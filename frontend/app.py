import streamlit as st
import requests
import datetime

# URL of the API endpoint you want to post to
API_URL = "http://127.0.0.1:8000/upload_image" # Replace with your actual API endpoint

st.set_page_config(layout="wide")

st.title("ShelfSence AI", 
         help="AI-powered shelf analysis for retail optimization")

# Create two columns of equal width
left_column, right_column = st.columns(2)

def call_backend_API(api_url, uploaded_file, store_id, aisle_id, timestamp):
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

def default_text_gray():
    st.markdown("""
        <style>
        .stTextInput input::placeholder {
            color: gray; /* Changes the placeholder text color to gray */
            opacity: 1;  /* Ensure full visibility across browsers */
        }
        </style>
        """, unsafe_allow_html=True)

def store_id_input_box():
    store_id = st.text_input("Store ID", placeholder="(Optional)")
    default_text_gray()
    if store_id == None : store_id = 0
    
    return store_id

def aisle_id_input_box():
    aisle_id = st.text_input("Aisle ID", placeholder="(Optional)")
    default_text_gray()
    if aisle_id == None : aisle_id = 0
    
    return aisle_id

def timestamp_input_box():
    upload_time = st.datetime_input(
        "Timestamp",
        value = None,    
    )
    if upload_time == None: 
        upload_time = datetime.datetime.now().strftime("%Y-%M-%D %H:%m")

    return upload_time

with left_column:
    ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]
    
    left_column1, left_column2, left_column3 = st.columns(3)
    
    with left_column1: store_id = store_id_input_box()
    with left_column2: aisle_id = aisle_id_input_box()       
    with left_column3: timestamp = timestamp_input_box()

    uploaded_file = st.file_uploader(
        "Upload image of Shelf (JPG/PNG/WEBP)", accept_multiple_files=False)

    if uploaded_file is not None:

        st.image(uploaded_file, caption="Uploaded Shelf Image", width="stretch")

        if st.button("Analyze Shelf"):
            with st.spinner("Analyzing..."):
                with right_column:
                    call_backend_API(API_URL, uploaded_file, store_id, aisle_id, timestamp)