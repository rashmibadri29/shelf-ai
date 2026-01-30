import logging
import os
from datetime import datetime
from typing import Annotated
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from services.shelf_analyzer import shelf_analyzer

logger = logging.getLogger(__name__)
logs_directory = '../logs'
if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)
logging.basicConfig(filename=os.path.join(logs_directory,'app.log'), level=logging.DEBUG,    
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

app = FastAPI()

# FastAPI's built-in way to handle cross-origin rules
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # It’s okay for requests to come from anywhere, 
    allow_methods=["*"], # using any HTTP method, 
    allow_headers=["*"], # with any headers
)

@app.post("/upload_image/")
async def analyze_shelf(store_id : Annotated[str, Form(description="Store ID")],  
                        aisle_id : Annotated[str, Form(description="Aisle ID")],  
                        timestamp : Annotated[datetime, Form(description="Timestamp")],   
                        file : Annotated[UploadFile, File(description="Image file to analyze")]  
                        ):
    
    # Read file contents in bytes
    content = await file.read()

    logging.info(f"Received File: {file.filename} \
                with Content Type: {file.content_type} \
                and Size: {len(content)} bytes")

    shelf_analysis = shelf_analyzer(store_id, aisle_id, timestamp, file)
    
    return shelf_analysis

