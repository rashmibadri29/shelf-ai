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

## Project Status
v1.0.0 — Stable API and UI with dummy inference.
Future versions will replace dummy logic with real multimodal models.

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