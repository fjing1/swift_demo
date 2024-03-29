import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError
from datetime import datetime
import utils

 

def upload_to_aws(aws_access_key, aws_secret_access,  local_file, bucket, s3_file):
    # get these from aws user
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key,
                      aws_secret_access_key=aws_secret_access)

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

def main(aws_access_key, aws_secret_access, bucket_name):
    st.title('Resume Uploader')
    st.write('Upload your resume in PDF format and we will store it in our AWS S3 Bucket!')

    first_name = st.text_input('First Name')
    last_name = st.text_input('Last Name')
    target_title = st.text_input('Target title, such as software developer, data scientist').lower()
    target_location = st.text_input("Target cities, such as Toronto, Vancouver, Montreal, etc").lower()
    email = st.text_input("Enter the email you want to receive job apply links").lower()
    # TODO: 
    # salary expectation
    # location
    # highest education?
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    if uploaded_file is not None and first_name and last_name:
        with open("temp_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        if st.button('Submit'):
            #timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            s3_filename = f"{first_name}_{last_name}_{target_title}_{target_location}_{email}.pdf"
            upload_to_aws( aws_access_key, aws_secret_access,'temp_file.pdf', bucket_name, s3_filename)
            st.success("Your resume has been uploaded successfully!")

if __name__ == "__main__":
    aws_access_key = utils.load_specific_api_key(filename='credential.txt', key_name='aws_access_key_id') 
    aws_secret_access = utils.load_specific_api_key(filename='credential.txt', key_name='aws_secret_access_key') 
    bucket_name = "swift-hire-felix-kelly-us-west-2"
    main(aws_access_key, aws_secret_access, bucket_name)
