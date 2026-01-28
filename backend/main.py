import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
logging.basicConfig(filename='../logs/app.log', level=logging.DEBUG,    
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
async def analyze_shelf(file : UploadFile = File(...)):

    # Read file contents in bytes
    content = await file.read()

    logging.info(f"Received File: {file.filename} \
                with Content Type: {file.content_type} \
                and Size: {len(content)} bytes")


    return { "image_path": file.filename,
        "status": "issues",
        "confidence": 0.87,
        "summary": {
            "products_detected": 6,
            "issues": ["missing", "misplaced", "overstock"]
        },
        "products": [
            {
            "name": "Coca-Cola 500ml",
            "status": "ok",
            "confidence": 0.94
            },
            {
            "name": "Pepsi 500ml",
            "status": "missing",
            "confidence": 0.81
            },
            {
            "name": "Sprite 500ml",
            "status": "misplaced",
            "confidence": 0.76
            },
            {
            "name": "Fanta 500ml",
            "status": "overstock",
            "confidence": 0.83
            }
        ]
        }

