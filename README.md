# Shelf AI – Multimodal Retail Shelf Analysis

A prototype AI application for retail shelf analysis using
FastAPI (backend) and Streamlit (frontend).

## Features
- Image upload
- Shelf compliance analysis (dummy inference)
- Product-level issue detection
- JSON-based API

## Architecture Diagram
Image <br>
 ↓ <br>
YOLOv8 (object detection) <br>
 ↓ <br>
CLIP (product matching) <br>
 ↓ <br>
Shelf rule engine <br>
 ↓ <br>
LLM reasoning <br>
 ↓ <br>
API (FastAPI) <br>
 ↓ <br>
UI (Streamlit) <br>

## Tech Stack
- Python
- FastAPI
- Streamlit
- YOLOv8 (vision model)
- Pytorch
- CLIP (image to text conversion model)
- Gemini (Google Gen-AI reasoning model)

## Project Status
v1.0.0 — Stable API and UI with dummy inference. <br>
v1.1.0 - Complete YOLO-CLIP inference pipeline with aggregation logic <br>
v1.2.0 - Shelf analysis pipeline (YOLO + CLIP + LLM) <br>

## Before you Run
Head to [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key) and generate an API key. <br>
To add GOOGLE_API_KEY to environment, follow these steps:
1. Create a .env file in shelfsense_ai directory
2. Enter the following line and save file
```bash
GOOGLE_API_KEY="<YOUR API KEY>"
```

## How to Run

### Backend
```bash
cd backend
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
streamlit run app.py 
```