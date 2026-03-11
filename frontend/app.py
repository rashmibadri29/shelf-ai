import streamlit as st
import requests
import datetime
import os
# URL of the API endpoint you want to post to
API_URL = "http://127.0.0.1:8000/upload_image" # Replace with your actual API endpoint

st.set_page_config(layout="wide")

st.title("ShelfSence AI", 
         help="AI-powered shelf analysis for retail optimization")

# Create two columns of equal width
left_column, right_column = st.columns(2)

# Display analysis results by category
def display_analysis_results(response):
    for key in response:
        if key in ["image_path", "store_id", "aisle_id", "timestamp"]:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.text(key + " :")
            with col2:
                st.text(response[key])
        else:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.text(key + " :")
            with col2:
                st.json(response[key])

# Call the backend API to run analysis and get response
def call_backend_API(api_url, uploaded_file, store_id, aisle_id, timestamp):
    data = {
                "store_id": store_id,
                "aisle_id": aisle_id,
                "timestamp": timestamp
            }
    files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file,
                    uploaded_file.type
                    )
            }
    response = requests.post(
                api_url,
                data = data,
                files = files
                )
    if response.status_code == 200:
        st.success("Analysis complete")
        display_analysis_results(response.json())
    else:
        st.error(f"Error: {response.status_code}")
        st.text(response.text)

# Change the default text color to gray
def default_text_gray():
    st.markdown("""
        <style>
        .stTextInput input::placeholder {
            color: gray; /* Changes the placeholder text color to gray */
            opacity: 1;  /* Ensure full visibility across browsers */
        }
        </style>
        """, unsafe_allow_html=True)

# Create text box for Store ID user input
def store_id_input_box():
    store_id = st.text_input("Store ID", placeholder="(Optional)")
    default_text_gray()
    if not store_id : store_id = "0"
    
    return store_id

# Create text box for Aisle ID user input
def aisle_id_input_box():
    aisle_id = st.text_input("Aisle ID", placeholder="(Optional)")
    default_text_gray()
    if not aisle_id : aisle_id = "0"
    
    return aisle_id

# Create text box for Timestamp user input
def timestamp_input_box():
    upload_time = st.datetime_input(
        "Timestamp",
        value = None,    
    )
    if upload_time == None: 
        upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    return upload_time

# Main loop
with left_column:    
    # Create three columns of equal width
    left_column1, left_column2, left_column3 = st.columns(3)
    
    # Each column contains a text box for Store ID, Aisle ID and Timestamp
    with left_column1: store_id = store_id_input_box()
    with left_column2: aisle_id = aisle_id_input_box()       
    with left_column3: timestamp = timestamp_input_box()

    # Create a file uploader that only accepts images
    uploaded_file = st.file_uploader(
        "Upload image of Shelf (JPG/PNG/WEBP)", accept_multiple_files=False)

    if uploaded_file is not None:
        # Image displayed on upload
        st.image(uploaded_file, caption="Uploaded Shelf Image", width="stretch")

        with right_column:
            # Analysis runs in the backend on button click
            if st.button("Analyze Shelf"):
                with st.spinner("Running analysis..."):
                        call_backend_API(API_URL, uploaded_file, store_id, aisle_id, timestamp)