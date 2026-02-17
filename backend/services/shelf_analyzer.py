import datetime
import os
from fastapi import UploadFile
import numpy as np
import cv2
from services.yolo_clip_inference import yolo_clip_inference
import configparser
import json

def uploadfile_to_cv2(upload_file):
    """
    Converts a FastAPI UploadFile to an OpenCV image (numpy array in BGR format).
    Args:        upload_file (UploadFile): The uploaded file from FastAPI.
    Returns:     image (np.ndarray): The decoded image in BGR format (OpenCV default).
    """
    upload_file.file.seek(0) # Ensure the file pointer is at the beginning of the file
    # Read raw bytes
    image_bytes = upload_file.file.read()

    # Convert bytes → numpy buffer
    np_arr = np.frombuffer(image_bytes, np.uint8)

    # Decode image
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return image  # BGR format (OpenCV default)

def analyze_yolo_clip_detections(PM_results): # part 1 of yolo clip detection analysis
    """
    Summarizes detected products on shelf by aggregating counts, average confidence,
    and max confidence for each detected product in the image.
    Args:        PM_results (list): A list of tuples containing product names and their confidence scores.
    Returns:     detection_analysis (list): A list of dictionaries summarizing the detected products with their counts and confidence metrics.
    """

    # Load shelf rules from JSON file
    with open('config/shelf_rules.json', 'r') as file:
        data = json.load(file)
    
    detection_analysis = [] # Initialize an empty list to hold the analysis results for each detected product

    # Aggregate counts and confidence scores for each detected product
    for product, confidence in PM_results:
        for item in detection_analysis:
            if item['product_name'] == product:
                item['count'] += 1
                item['list_of_confidence'].append(round(confidence, 2))
                break
        else:
            detection_analysis.append({"product_name": product, "count": 1, "list_of_confidence": [round(confidence, 2)]})
    
    # Calculate average and max confidence for each detected product
    for product in detection_analysis:
        product['avg_confidence'] = round(sum([float(c) for c in product['list_of_confidence']]) / len(product['list_of_confidence']), 2)
        product['max_confidence'] = max(product['list_of_confidence'])  
        
        if data[product["product_name"]]["min"] <= product["count"] <= data[product["product_name"]]["max"]:
            product["status"] = "ok"
        elif product["count"] < data[product["product_name"]]["min"]:
            product["status"] = "low-stock"
        else: product["status"] = "overstock"

    return detection_analysis

def summarize_yolo_clip_detections(detection_analysis): # part 2 of yolo clip detection analysis
    '''
    Summarizes the overall shelf status based on the analysis of detected products, 
    including total products detected, issues identified, and the ratio of unknown products.
    Args:        detection_analysis (list): A list of dictionaries containing analysis results for each detected product.
    Returns:     detection_summary (dict): A summary of the overall shelf status including total products, issues, and unknown product ratio.
    '''
    # Analyze the product issues and calculate the ratio of unknown products
    product_issues = []
    len_of_unknowns = 0
    for product in detection_analysis:
        if product["status"] not in product_issues:
            product_issues.append(product["status"])
        if product["product_name"] == "Unknown Product":
            len_of_unknowns += 1
    # Generate a summary of the overall shelf status based on the analysis of detected products
    detection_summary = {
        "total_products_detected": len(detection_analysis),
        "issues": list(set(product["status"] for product in detection_analysis)),
        "unknown ratio": len_of_unknowns / len(detection_analysis) if detection_analysis else 0,
        "status": "ok" if all(issue == "ok" for issue in product_issues) else "issues"
    }
    
    return detection_summary

def shelf_analyzer(store_id: str, aisle_id: str, timestamp: datetime, file: UploadFile):
    """
    Analyze the shelf image and return the analysis results.
    Args:
        store_id (str): Store ID
        aisle_id (str): Aisle ID
        timestamp (datetime): Timestamp of the image
        file (UploadFile): Uploaded image file
    Returns:
        analysis_result (dict): Analysis results including detected products and their statuses
    """
    img = uploadfile_to_cv2(file)  # Convert uploaded file to OpenCV image format (BGR)

    # Perform YOLO + CLIP inference to detect products (using YOLO model) and get their matches (using CLIP embeddings)
    detected_products, product_matches = yolo_clip_inference(img)

    # Analyze the detected products and summarize the shelf status based on the analysis
    detection_analysis = analyze_yolo_clip_detections(product_matches)
    detection_summary = summarize_yolo_clip_detections(detection_analysis)

    # Generate a new filename for the uploaded image based on store ID, aisle ID, and timestamp
    new_filename = f"{store_id}_{aisle_id}_{timestamp.strftime('%Y%m%d_%H%M')}_{os.path.splitext(file.filename)[-1]}"

    analysis_result = {
        "image_path": new_filename,
        "store_id": store_id,
        "aisle_id": aisle_id,
        "timestamp": timestamp.isoformat(),
        "summary": detection_summary,
        "products": detection_analysis
    }
    

    return analysis_result
