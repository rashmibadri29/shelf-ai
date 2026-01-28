import streamlit as st
import requests

# URL of the API endpoint you want to post to
API_URL = "http://127.0.0.1:8000/upload_image" # Replace with your actual API endpoint

st.set_page_config(layout="wide")

st.title("ShelfSence AI", 
         help="AI-powered shelf analysis for retail optimization")

# Create two columns of equal width
left_column, right_column = st.columns(2)

with left_column:
    ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]
    
    uploaded_file = st.file_uploader(
        "Upload image (JPG/PNG/WEBP)", accept_multiple_files=False)

    if uploaded_file is not None:

        st.image(uploaded_file, caption="Uploaded Image", width="stretch")

        if st.button("Analyze Shelf"):
            with st.spinner("Analyzing..."):
                with right_column:
                    response = requests.post(
                        API_URL,
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file,
                                uploaded_file.type
                            )
                        }
                    )

                    if response.status_code == 200:
                        st.success("Analysis complete")
                        st.json(response.json())
                    else:
                        st.error(f"Error: {response.status_code}")
                        st.text(response.text)