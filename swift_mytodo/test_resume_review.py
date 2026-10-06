"""Tests for resume_review.py (stdlib unittest, no LLM or network needed).

Run from swift_mytodo/:
    python -m unittest test_resume_review -v
"""
import unittest

import resume_review as rr


SAMPLE = """Jane Doe
Austin, TX | jane@example.com | github.com/janedoe
Technical Skills
Languages: Python, Java, SQL
Technologies: AWS(EKS, S3), Kubernetes, Kafka, Terraform
Experience
Site Reliability Engineer (Onsite) Austin, TX
Acme www.acme.com Jul 2024 - Present
• Responsible for the deployment pipeline and on-call rotation for various internal services running on Kubernetes clusters
in two regions
• Migrated 40 services to Terraform, cutting provisioning time from 2 days to 20 minutes
• Optimized Python alerting jobs, demonstrating strong problem-solving skills
Software Engineer Intern (Remote) Toronto, ON
Beta Corp Jan 2023 - Aug 2023
• Worked on reccomendation APIs in Java
"""


class ParseTests(unittest.TestCase):
    def test_wrapped_bullet_is_joined_and_role_line_is_not(self):
        bullets, _ = rr.parse_resume(SAMPLE)
        texts = [b[1] for b in bullets]
        self.assertEqual(len(texts), 4)
        self.assertTrue(texts[0].endswith("in two regions"))
        self.assertNotIn("Intern", texts[2])

    def test_role_context_strips_url_and_dates(self):
        bullets, _ = rr.parse_resume(SAMPLE)
        self.assertEqual(bullets[0][0], "Acme")
        self.assertEqual(bullets[3][0], "Beta Corp")

    def test_skills_drop_labels_and_parentheticals(self):
        _, sections = rr.parse_resume(SAMPLE)
        self.assertEqual(rr.extract_skills(sections),
                         ["Python", "Java", "SQL", "AWS", "Kubernetes", "Kafka", "Terraform"])


class BulletRuleTests(unittest.TestCase):
    def kinds(self, text):
        return {i.kind for i in rr.review_bullet(1, "", text).issues}

    def test_strong_quantified_bullet_is_clean(self):
        self.assertEqual(self.kinds(
            "Migrated 40 services to Terraform, cutting provisioning time from 2 days to 20 minutes"), set())

    def test_weak_opener_and_vague(self):
        kinds = self.kinds("Responsible for the deployment pipeline for various internal services")
        self.assertTrue({"weak-opener", "vague", "no-metric"} <= kinds)

    def test_filler_is_flagged_and_removed_in_suggestion(self):
        review = rr.review_bullet(1, "", "Optimized Python alerting jobs, demonstrating strong problem-solving skills")
        self.assertIn("filler", {i.kind for i in review.issues})
        self.assertNotIn("demonstrating", review.suggestion)
        self.assertIn("[result:", review.suggestion)

    def test_typos_and_glued_words_are_fixed(self):
        review = rr.review_bullet(1, "", "Rebuilt the reccomendation service, generating 2M USDin revenue per year")
        self.assertIn("typo", {i.kind for i in review.issues})
        self.assertIn("recommendation", review.suggestion)
        self.assertIn("USD in", review.suggestion)

    def test_acronym_plurals_are_not_glued_words(self):
        self.assertNotIn("typo", self.kinds("Designed 12 REST APIs and GPUs scheduling, reducing latency by 30%"))

    def test_pronoun_and_length(self):
        self.assertIn("pronoun", self.kinds("I built our CI system that cut build time by 50% for 8 teams"))
        self.assertIn("too-short", self.kinds("Wrote Python scripts"))


class ReportTests(unittest.TestCase):
    def test_doc_level_findings(self):
        report = rr.review_resume(SAMPLE)
        doc = {i.kind: i.message for i in report.doc_issues}
        self.assertIn("unproven-skills", doc)
        self.assertIn("SQL", doc["unproven-skills"])
        self.assertIn("Kafka", doc["unproven-skills"])
        self.assertNotIn("Kubernetes,", doc["unproven-skills"])
        self.assertNotIn("no-github", doc)  # link is in the contact header
        self.assertEqual(report.bullet_count, 4)
        self.assertEqual(report.quantified_count, 2)
        self.assertTrue(0 <= report.score <= 100)
        self.assertTrue(report.top_fixes)

    def test_no_bullets(self):
        report = rr.review_resume("John Smith\nExperience\nI did lots of things at a company.")
        self.assertIn("no-bullets", {i.kind for i in report.doc_issues})

    def test_format_report_renders(self):
        text = rr.format_report(rr.review_resume(SAMPLE))
        self.assertIn("RESUME REVIEW", text)
        self.assertIn("Suggested:", text)


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def batch(self, prompts, config=None):
        assert len(prompts) == len(self.replies)
        return self.replies


class LLMLayerTests(unittest.TestCase):
    def test_rewrites_with_invented_numbers_are_rejected(self):
        report = rr.review_resume(SAMPLE)
        flagged = [r for r in report.bullets if r.issues]
        replies = ["Owned the deployment pipeline, cutting deploy time by 63%"]  # invented number
        replies += ["- Built recommendation APIs in Java, serving [N] requests/day"] * (len(flagged) - 1)
        replies += ["Overall summary."]
        self.assertIsNone(rr.add_llm_rewrites(report, llm=FakeLLM(replies)))
        self.assertEqual(flagged[0].llm_rewrite, "")
        self.assertEqual(flagged[-1].llm_rewrite, "Built recommendation APIs in Java, serving [N] requests/day")
        self.assertEqual(report.llm_summary, "Overall summary.")

    def test_unreachable_llm_degrades_gracefully(self):
        class Down:
            def batch(self, prompts, config=None):
                raise ConnectionError("refused")

        report = rr.review_resume(SAMPLE)
        note = rr.add_llm_rewrites(report, llm=Down())
        self.assertIn("LLM unavailable", note)
        self.assertIn("RESUME REVIEW", rr.format_report(report))


if __name__ == "__main__":
    unittest.main()


class ClaudeProviderTests(unittest.TestCase):
    """utils.get_llm() wiring for Claude, with the Anthropic client mocked out."""

    def setUp(self):
        from unittest import mock
        import utils

        self.utils = utils
        self.client = mock.MagicMock()
        patcher = mock.patch.object(utils, "_anthropic_client", return_value=self.client)
        patcher.start()
        self.addCleanup(patcher.stop)
        provider = mock.patch.object(utils, "LLM_PROVIDER", "anthropic")
        provider.start()
        self.addCleanup(provider.stop)

    def reply(self, text, stop_reason="end_turn"):
        from types import SimpleNamespace as NS

        self.client.beta.messages.create.return_value = NS(
            content=[NS(type="thinking", thinking=""), NS(type="text", text=text)],
            stop_reason=stop_reason,
            stop_details=NS(category="cyber") if stop_reason == "refusal" else None,
        )

    def test_request_shape_and_text_extraction(self):
        self.reply("Led 3 DR drills")
        out = self.utils.get_llm().invoke("rewrite this")
        self.assertEqual(out.content, "Led 3 DR drills")
        kwargs = self.client.beta.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], self.utils.ANTHROPIC_MODEL)
        self.assertEqual(kwargs["fallbacks"], "default")
        self.assertEqual(kwargs["betas"], ["server-side-fallback-2026-07-01"])
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "rewrite this"}])
        self.assertNotIn("temperature", kwargs)

    def test_fallbacks_can_be_disabled(self):
        from unittest import mock

        self.reply("ok")
        with mock.patch.object(self.utils, "ANTHROPIC_FALLBACKS", False):
            self.utils.get_llm().invoke("hi")
        kwargs = self.client.beta.messages.create.call_args.kwargs
        self.assertNotIn("fallbacks", kwargs)
        self.assertNotIn("betas", kwargs)

    def test_composes_in_langchain_chain(self):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        self.reply("ok")
        chain = ChatPromptTemplate.from_template("Q: {q}") | self.utils.get_llm() | StrOutputParser()
        self.assertEqual(chain.invoke({"q": "hi"}), "ok")
        self.assertIn("Q: hi", self.client.beta.messages.create.call_args.kwargs["messages"][0]["content"])

    def test_refusal_raises_and_reviewer_degrades(self):
        self.reply("", stop_reason="refusal")
        report = rr.review_resume(SAMPLE)
        note = rr.add_llm_rewrites(report, llm=self.utils.get_llm())
        self.assertIn("declined", note)
