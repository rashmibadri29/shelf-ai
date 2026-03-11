# Shelf AI – Multimodal Retail Shelf Analysis

A prototype AI application for retail shelf analysis using
FastAPI (backend) and Streamlit (frontend).

## Features
- Image upload
- Shelf compliance analysis (dummy inference)
- Product-level issue detection
- JSON-based API

## Tech Stack
- Python
- FastAPI
- Streamlit
- YOLOv8 (vision model)
- CLIP (image to text conversion model)
- Gemini (Google Gen-AI reasoning model)

## Project Status
v1.0.0 — Stable API and UI with dummy inference.
v1.1.0 - Complete YOLO-CLIP inference pipeline with aggregation logic

## Before you Run
Head to [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key) and generate an API key. 
Add GOOGLE API KEY to environment variable by running script:
```bash
python add_google_api_key.py
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