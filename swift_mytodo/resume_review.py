"""Resume reviewer: small, concrete edits that help engineers present themselves.

Many strong engineers undersell their work on paper: bullets say what they were
*near* ("Contributed to", "Worked on") instead of what they *did*, results are
missing or vague ("significantly improved"), soft-skill filler crowds out
evidence, and skills listed at the top never show up in the experience below.
This module reviews a resume bullet by bullet and suggests minimal fixes. It
never rewrites the whole resume and never invents facts.

Two layers:
  - rule-based review (default): fast, deterministic, no LLM or network needed.
    Flags weak openers, missing metrics, vague wording, filler, pronouns,
    length, common typos, overused verbs, and skills that lack evidence. Applies
    safe mechanical edits (typo fixes, filler removal, metric placeholders).
  - --llm: additionally asks the LLM from utils.get_llm() (Claude by default,
    or local Ollama with LLM_PROVIDER=ollama) for a minimal rewrite of each
    flagged bullet. Rewrites that introduce numbers not in the original are
    rejected, so the model can't make up achievements; missing metrics come
    back as [placeholders] for the candidate to fill in.

Examples:
    python resume_review.py ../Fei_Jing_Resume_2026.pdf
    python resume_review.py resume.txt --top 5
    python resume_review.py ../Fei_Jing_Resume_2026.pdf --llm
    python resume_review.py ../Fei_Jing_Resume_2026.pdf --json

Kept free of the heavy LangChain stack at import time so web.py can use it.
"""
import os
import re
import json
import argparse
from collections import Counter
from dataclasses import dataclass, field, asdict


# --- Word lists ---------------------------------------------------------------
# Openers that describe proximity to work rather than ownership of it, mapped
# to stronger verbs the candidate can pick from *if accurate*.
WEAK_OPENERS = {
    "responsible for": "Owned / Ran / Managed",
    "contributed to": "Led / Drove / Delivered / Built",
    "helped": "Built / Enabled / Delivered",
    "helped to": "Built / Enabled / Delivered",
    "assisted": "Supported / Built / Delivered",
    "assisted with": "Supported / Built / Delivered",
    "assisted in": "Supported / Built / Delivered",
    "worked on": "Built / Developed / Shipped",
    "worked with": "Partnered with / Integrated",
    "participated in": "Drove / Delivered / Ran",
    "involved in": "Drove / Delivered / Built",
    "was involved in": "Drove / Delivered / Built",
    "tasked with": "Owned / Delivered",
    "duties included": "Owned / Ran",
    "in charge of": "Owned / Led",
    "utilized": "Used / Applied",
    "leveraged": "Used / Applied",
    "was part of": "Delivered / Built",
}

# Soft-skill claims that take space without evidence. Recruiters skip them;
# the bullet should *show* the skill through a result instead.
FILLER_PATTERNS = [
    r",?\s*demonstrating (?:strong |excellent |great )?[\w\s,-]*?skills",
    r",?\s*showcasing (?:strong |excellent |great )?[\w\s,-]*?skills",
    r"\bteam player\b",
    r"\bdetail[- ]oriented\b",
    r"\bhard[- ]working\b",
    r"\bself[- ]starter\b",
    r"\bresults[- ]driven\b",
    r"\bpassionate about\b",
    r"\bgo[- ]getter\b",
    r"\bexcellent communication skills\b",
    r"\bthink outside the box\b",
    r"\bsynergy\b",
]

# Words that gesture at impact without stating it.
VAGUE_WORDS = [
    "significant", "significantly", "various", "numerous", "several",
    "many", "multiple", "a lot of", "lots of", "huge", "greatly",
    "substantially", "dramatically", "millions", "thousands", "etc",
]

# Signals that a bullet states a result, not just an activity.
RESULT_WORDS = [
    "reduc", "increas", "improv", "sav", "cut", "grew", "grow", "boost",
    "decreas", "eliminat", "accelerat", "lower", "rais", "prevent",
    "enabl", "unblock", "achiev", "outperform", "from",
]

PRONOUNS = r"\b(?:I|me|my|we|our|us)\b"

# Small, high-precision misspelling list (a full spellchecker misfires on the
# split words PDF extraction produces, e.g. "W aterloo").
COMMON_TYPOS = {
    "reccomendation": "recommendation", "recomendation": "recommendation",
    "reccommendation": "recommendation", "acheive": "achieve",
    "acheived": "achieved", "recieve": "receive", "recieved": "received",
    "seperate": "separate", "occured": "occurred", "sucessful": "successful",
    "succesful": "successful", "sucessfully": "successfully",
    "succesfully": "successfully", "managment": "management",
    "enviroment": "environment", "developement": "development",
    "maintainance": "maintenance", "maintenence": "maintenance",
    "perfomance": "performance", "performace": "performance",
    "infrastucture": "infrastructure", "accross": "across",
    "dependancy": "dependency", "dependancies": "dependencies",
    "efficency": "efficiency", "effecient": "efficient",
    "scaleable": "scalable", "begining": "beginning", "untill": "until",
    "wich": "which", "teh": "the", "responsability": "responsibility",
    "optimzed": "optimized", "optimised": "optimized",
    "implmented": "implemented", "implemeted": "implemented",
    "architecure": "architecture", "databse": "database",
    "deployement": "deployment", "analysys": "analysis",
    "collaborted": "collaborated", "automatation": "automation",
}

SECTION_NAMES = {
    "education", "experience", "workexperience", "professionalexperience",
    "employment", "skills", "technicalskills", "projects", "personalprojects",
    "publications", "summary", "profile", "certifications", "awards",
    "leadership", "volunteer", "volunteering", "interests", "activities",
}
SKILL_SECTIONS = {"skills", "technicalskills"}

BULLET_RE = re.compile(r"^\s*(?:[•●▪■◦‣∙·*–-]|o\s)\s*")
DATE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{4}"
    r"|\b(?:19|20)\d{2}\s*[-–]\s*(?:(?:19|20)\d{2}|present)\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"\b(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)+(?:/\S*)?", re.IGNORECASE)
METRIC_RE = re.compile(r"\d|\$|%|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|dozen|hundred)\b",
                       re.IGNORECASE)
# A role/title line, e.g. "Software Engineer Intern (Hybrid) Sunnyvale, CA".
ROLE_LINE_RE = re.compile(r",\s*[A-Z]{2}\s*$|\((?i:on-?site|remote|hybrid)\)|(?i:\bremote)\s*$")
# PDF text often glues words together, e.g. "USDin", "minsper" (see GLUED_RE uses).
GLUED_RE = re.compile(r"\b([A-Z]{2,})([a-rt-z][a-z]+)\b")

MAX_WORDS = 35
MIN_WORDS = 7


@dataclass
class Issue:
    kind: str
    message: str
    penalty: int


@dataclass
class BulletReview:
    index: int
    context: str
    original: str
    issues: list = field(default_factory=list)
    suggestion: str = ""
    llm_rewrite: str = ""
    score: int = 100


@dataclass
class ReviewReport:
    source: str
    score: int
    bullet_count: int
    quantified_count: int
    top_fixes: list
    doc_issues: list
    bullets: list
    llm_summary: str = ""

    def to_dict(self):
        return asdict(self)


# --- Parsing ------------------------------------------------------------------
def extract_text(source):
    """Return resume text from a PDF/text path or a file-like PDF upload."""
    is_path = isinstance(source, (str, os.PathLike))
    if is_path and not str(source).lower().endswith(".pdf"):
        with open(source, encoding="utf-8") as f:
            return f.read()

    from pypdf import PdfReader

    if is_path and not os.path.exists(source):
        raise FileNotFoundError(f"Resume not found: {source}")
    reader = PdfReader(source)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _norm_header(line):
    return re.sub(r"[^a-z]", "", line.lower())


def _role_context(line):
    """Shorten a header line like 'Apple www.apple.com Jul 2024 - Present' to 'Apple'."""
    line = DATE_RE.split(line)[0]
    line = URL_RE.sub("", line)
    line = re.sub(r"\s{2,}", " ", line).strip(" -–|,")
    return line[:60]


def parse_resume(text):
    """Split resume text into bullets (with their role context) and sections.

    PDF extraction hard-wraps long bullets, so a non-bullet line is treated as
    a continuation when the line before it ran close to the page width.

    :return: (bullets, sections) where bullets is a list of (context, text,
             section) tuples and sections maps a normalized section name to
             its raw lines.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    wrap_width = 0.8 * max((len(ln) for ln in lines), default=0)

    bullets, sections = [], {}
    section, context = "", ""
    current, prev_len = None, 0

    def flush():
        nonlocal current
        if current:
            bullets.append((context, " ".join(current), section))
        current = None

    for line in lines:
        header = _norm_header(line)
        if header in SECTION_NAMES:
            flush()
            section = header
            sections.setdefault(section, [])
        elif BULLET_RE.match(line):
            flush()
            current = [BULLET_RE.sub("", line).strip()]
        elif (current is not None and prev_len >= wrap_width
              and not DATE_RE.search(line) and not ROLE_LINE_RE.search(line)):
            current.append(line.strip())
        else:
            flush()
            if DATE_RE.search(line) or not context:
                context = _role_context(line) or context
        sections.setdefault(section, []).append(line)
        prev_len = len(line)
    flush()
    return bullets, sections


def extract_skills(sections):
    """Pull the comma-separated skills out of the skills section(s)."""
    raw = ", ".join(
        ln for name in SKILL_SECTIONS for ln in sections.get(name, [])
        if _norm_header(ln) not in SECTION_NAMES
    )
    raw = re.sub(r"\([^)]*\)", "", raw)          # drop "(EKS, ECR, ...)" details
    raw = re.sub(r"[A-Za-z][A-Za-z &/]*:", ",", raw)  # drop "Languages:" labels
    skills = []
    for item in re.split(r"[,;|•]", raw):
        item = re.sub(r"\s+", " ", item).strip(" .")
        if item and len(item) <= 30 and item.lower() not in {s.lower() for s in skills}:
            skills.append(item)
    return skills


# --- Rule-based review --------------------------------------------------------
def _contains_word(text, phrase):
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])",
                     text, re.IGNORECASE)


def _opener(text):
    lowered = text.lower()
    for phrase in sorted(WEAK_OPENERS, key=len, reverse=True):
        if lowered.startswith(phrase + " ") or lowered == phrase:
            return phrase
    return None


def _apply_safe_edits(text):
    """Mechanical, meaning-preserving fixes: typos, glued words, filler."""
    def fix_typo(m):
        fixed = COMMON_TYPOS[m.group(0).lower()]
        return fixed.capitalize() if m.group(0)[0].isupper() else fixed

    text = re.sub(r"\b(" + "|".join(COMMON_TYPOS) + r")\b", fix_typo, text, flags=re.IGNORECASE)
    text = GLUED_RE.sub(r"\1 \2", text)
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip(" ,;")


def review_bullet(index, context, text):
    review = BulletReview(index=index, context=context, original=text)
    issues = review.issues
    words = text.split()

    opener = _opener(text)
    if opener:
        issues.append(Issue(
            "weak-opener",
            f'Starts with "{text[:len(opener)]}", which says you were near the work, '
            f"not what you did. If accurate, try: {WEAK_OPENERS[opener]}.",
            20,
        ))

    has_metric = bool(METRIC_RE.search(text))
    has_result = any(w in text.lower() for w in RESULT_WORDS)
    if not has_metric:
        issues.append(Issue(
            "no-metric",
            "No number anywhere. Add scale or impact you can stand behind: time saved, "
            "% faster/cheaper, requests/day, users, services, incidents avoided.",
            30 if not has_result else 20,
        ))
    elif not has_result:
        issues.append(Issue(
            "no-result",
            "Describes the activity but not the outcome. Close with what changed "
            "because of it (e.g. ', cutting X from A to B').",
            10,
        ))

    vague = [w for w in VAGUE_WORDS if _contains_word(text, w)]
    if vague:
        issues.append(Issue(
            "vague",
            f"Vague wording ({', '.join(vague)}). Replace with the actual figure "
            f"or drop it.",
            10,
        ))

    filler = [m.group(0).strip(" ,") for p in FILLER_PATTERNS
              for m in [re.search(p, text, re.IGNORECASE)] if m]
    if filler:
        issues.append(Issue(
            "filler",
            f'Soft-skill filler ("{filler[0]}"). Cut it and let the result show '
            f"the skill.",
            15,
        ))

    if re.search(PRONOUNS, text):
        issues.append(Issue("pronoun", "Drop first-person pronouns (I/my/we/our).", 10))

    if len(words) > MAX_WORDS:
        issues.append(Issue(
            "too-long",
            f"{len(words)} words. Keep bullets to one idea of roughly 15-30 words; "
            f"split or trim.",
            10,
        ))
    elif len(words) < MIN_WORDS:
        issues.append(Issue(
            "too-short",
            "Too thin to show impact. Add what you built, with what, and the result.",
            15,
        ))

    typos = sorted({m.group(0) for m in re.finditer(
        r"\b(" + "|".join(COMMON_TYPOS) + r")\b", text, re.IGNORECASE)})
    glued = [m.group(0) for m in GLUED_RE.finditer(text)]
    if typos or glued:
        found = ", ".join(typos + glued)
        issues.append(Issue("typo", f"Spelling/spacing: {found}.", 5 * len(typos + glued)))

    suggestion = _apply_safe_edits(text)
    if not has_metric:
        suggestion = suggestion.rstrip(".") + ", [result: e.g. reducing X by N% / saving N hrs/week]"
    if opener:
        suggestion = f"[{WEAK_OPENERS[opener].split(' / ')[0]}?]" + suggestion[len(opener):]
    review.suggestion = suggestion if suggestion != text else ""
    review.score = max(0, 100 - sum(i.penalty for i in issues))
    return review


def _doc_issues(text, bullets, sections, reviews):
    issues = []

    if not bullets:
        issues.append(Issue(
            "no-bullets",
            "No bullet points found. Recruiters skim; list each role's work as "
            "3-5 short bullets that each end in a result.",
            15,
        ))
        return issues

    openers = Counter(b[1].split()[0].lower().strip(",") for b in bullets if b[1].split())
    repeated = [f"{w.capitalize()} x{n}" for w, n in openers.most_common() if n >= 3]
    if repeated:
        issues.append(Issue(
            "repeated-verbs",
            f"Same opening verb used repeatedly ({', '.join(repeated)}). Vary them "
            f"so each bullet reads as a distinct contribution (Designed, Migrated, "
            f"Automated, Cut, Scaled, Debugged, Shipped...).",
            3 * len(repeated),
        ))

    skills = extract_skills(sections)
    evidence = " ".join(b[1] for b in bullets if b[2] not in SKILL_SECTIONS)
    unproven = [s for s in skills if not _contains_word(evidence, s)]
    if skills and unproven:
        issues.append(Issue(
            "unproven-skills",
            f"Listed in Skills but never shown in your experience: {', '.join(unproven)}. "
            f"If you've used them for real, mention them in a bullet where they "
            f"mattered; otherwise consider dropping them.",
            min(10, 2 * len(unproven)),
        ))

    contact = "\n".join(text.splitlines()[:6])
    if not (re.search(r"(?:github|gitlab)\.com", text, re.IGNORECASE)
            or re.search(r"github|gitlab|portfolio", contact, re.IGNORECASE)):
        issues.append(Issue(
            "no-github",
            "No GitHub/portfolio link. For engineers, a link to real code is cheap "
            "credibility (only if it's presentable).",
            3,
        ))

    quantified = sum(1 for r in reviews if not any(i.kind == "no-metric" for i in r.issues))
    if quantified / len(reviews) < 0.5:
        issues.append(Issue(
            "low-quantification",
            f"Only {quantified}/{len(reviews)} bullets have a number. Aim for at "
            f"least half. Even rough, honest scale (\"~40 services\", \"5 teams\") helps.",
            5,
        ))
    return issues


def _top_fixes(reviews, doc_issues, limit=3):
    """The few changes most worth the candidate's time, in priority order."""
    counts = Counter(i.kind for r in reviews for i in r.issues)
    advice = {
        "no-metric": "Add a number to {n} bullet(s) with no measurable result.",
        "weak-opener": "Rewrite {n} bullet(s) that open with passive phrasing to lead with what you did.",
        "filler": "Cut soft-skill filler from {n} bullet(s); show the skill with a result instead.",
        "vague": "Swap vague words for actual figures in {n} bullet(s).",
        "typo": "Fix spelling/spacing in {n} bullet(s).",
        "too-long": "Trim or split {n} overlong bullet(s).",
        "no-result": "State the outcome in {n} bullet(s) that only list the activity.",
    }
    weighted = sorted(
        ((kind, n) for kind, n in counts.items() if kind in advice),
        key=lambda kn: -kn[1] * max(i.penalty for r in reviews for i in r.issues if i.kind == kn[0]),
    )
    fixes = [advice[kind].format(n=n) for kind, n in weighted]
    fixes += [i.message.split(". ")[0] + "." for i in sorted(doc_issues, key=lambda i: -i.penalty)
              if i.kind in {"unproven-skills", "repeated-verbs", "no-bullets"}]
    return fixes[:limit]


def review_resume(text, source="resume"):
    """Run the rule-based review on resume text and return a ReviewReport."""
    bullets, sections = parse_resume(text)
    reviews = [review_bullet(i, ctx, body) for i, (ctx, body, _sec) in enumerate(bullets, 1)]
    doc_issues = _doc_issues(text, bullets, sections, reviews)

    bullet_avg = sum(r.score for r in reviews) / len(reviews) if reviews else 50
    score = int(max(0, min(100, round(bullet_avg - sum(i.penalty for i in doc_issues)))))
    quantified = sum(1 for r in reviews if not any(i.kind == "no-metric" for i in r.issues))

    return ReviewReport(
        source=source,
        score=score,
        bullet_count=len(reviews),
        quantified_count=quantified,
        top_fixes=_top_fixes(reviews, doc_issues),
        doc_issues=doc_issues,
        bullets=reviews,
    )


# --- Optional LLM layer -------------------------------------------------------
_REWRITE_PROMPT = """You are an expert technical recruiter editing one resume bullet for a software engineer.
Make the SMALLEST edit that fixes the listed problems. Rules:
- Keep every fact. Do NOT invent numbers, tools, scope or results.
- If a metric is missing, insert a short placeholder in square brackets, e.g. [X%] or [N hrs/week].
- Lead with a strong action verb; keep the original tense.
- Fix spelling and grammar. One sentence, at most 30 words.
- Return ONLY the rewritten bullet, no quotes or explanation.

Role: {context}
Problems: {problems}
Bullet: {bullet}
Rewritten bullet:"""

_SUMMARY_PROMPT = """You are an expert technical recruiter. Below are a software engineer's resume bullets and
an automated review. In 3 short bullet points, tell the candidate how this resume currently comes across
and the most important change to make it present their real level of ownership and impact. Be specific to
their content, encouraging, and do not invent facts.

Bullets:
{bullets}

Automated findings:
{findings}
"""


def _numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _clean_llm_line(text):
    text = text.strip().splitlines()[0] if text.strip() else ""
    return BULLET_RE.sub("", text).strip().strip('"').strip()


def add_llm_rewrites(report, llm=None):
    """Fill in llm_rewrite for flagged bullets and an overall llm_summary.

    Any rewrite introducing a number absent from the original is discarded.

    :return: an error string if the LLM is unreachable, else None.
    """
    if llm is None:
        import utils  # heavy LangChain import, only needed for --llm
        llm = utils.get_llm()

    flagged = [r for r in report.bullets if r.issues]
    prompts = [_REWRITE_PROMPT.format(
        context=r.context or "n/a",
        problems="; ".join(i.message for i in r.issues),
        bullet=r.original,
    ) for r in flagged]
    prompts.append(_SUMMARY_PROMPT.format(
        bullets="\n".join(f"- {r.original}" for r in report.bullets),
        findings="\n".join(report.top_fixes + [i.message for i in report.doc_issues]),
    ))

    try:
        # One request per bullet, sent concurrently.
        replies = [getattr(reply, "content", reply)
                   for reply in llm.batch(prompts, config={"max_concurrency": 4})]
    except Exception as exc:  # missing API key, connection refused, model not pulled, etc.
        return f"LLM unavailable ({type(exc).__name__}: {exc}). Showing rule-based review only."

    for review, reply in zip(flagged, replies):
        rewrite = _clean_llm_line(reply)
        if rewrite and _numbers(rewrite) <= _numbers(review.original):
            review.llm_rewrite = rewrite
    report.llm_summary = replies[-1].strip()
    return None


# --- Output -------------------------------------------------------------------
def format_report(report, top=None):
    out = [
        f"================ RESUME REVIEW: {report.source} ================",
        f"Score: {report.score}/100   "
        f"({report.bullet_count} bullets, {report.quantified_count} with a number)",
        "",
        "Top fixes:",
    ]
    out += [f"  {n}. {fix}" for n, fix in enumerate(report.top_fixes, 1)] or ["  (none, looks solid)"]

    if report.llm_summary:
        out += ["", "How it reads (LLM):", report.llm_summary]

    if report.doc_issues:
        out += ["", "Whole-resume notes:"]
        out += [f"  - {i.message}" for i in report.doc_issues]

    flagged = sorted((r for r in report.bullets if r.issues), key=lambda r: r.score)
    if top:
        flagged = flagged[:top]
    out += ["", f"Bullet-by-bullet ({len(flagged)} to improve, weakest first):"]
    for r in flagged:
        where = f" [{r.context}]" if r.context else ""
        out += ["", f"#{r.index}{where} (score {r.score})", f"  Original:  {r.original}"]
        out += [f"  - {i.message}" for i in r.issues]
        if r.llm_rewrite:
            out.append(f"  Rewrite:   {r.llm_rewrite}")
        elif r.suggestion:
            out.append(f"  Suggested: {r.suggestion}")
    out += ["", "Placeholders in [brackets] are for you to fill with real figures. "
                "Never put a number on your resume you can't explain in an interview."]
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Review a resume and suggest small edits that present the candidate better."
    )
    parser.add_argument("resume", help="Path to the resume (PDF, or .txt/.md).")
    parser.add_argument("--llm", action="store_true",
                        help="Also get minimal LLM rewrites of flagged bullets "
                             "(Claude by default; LLM_PROVIDER=ollama for local).")
    parser.add_argument("--top", type=int, help="Only show the N weakest bullets.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    cli = parser.parse_args()

    report = review_resume(extract_text(cli.resume), source=os.path.basename(cli.resume))
    note = add_llm_rewrites(report) if cli.llm else None

    if cli.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        if note:
            print(f"[review] {note}\n")
        print(format_report(report, top=cli.top))
