import datetime
import os
from fastapi import UploadFile
import numpy as np
import cv2
from services.yolo_clip_inference import yolo_clip_inference, display_results
from services.gemini_reasoning import perform_llm_reasoning
import configparser
import json
import time

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
    for product, confidence, BBox in PM_results:
        for item in detection_analysis:
            if item['product_name'] == product:
                item['count'] += 1
                item['on-shelf_availability'].append([BBox,round(confidence, 2)])
                break
        else:
            detection_analysis.append({"product_name": product, "count": 1, "on-shelf_availability": [[BBox,round(confidence, 2)]]})
    
    # Calculate average and max confidence for each detected product
    for product in detection_analysis:
        product['avg_confidence'] = round(sum([float(item[-1]) for item in product["on-shelf_availability"]]) / len(product["on-shelf_availability"]), 2)
        product['max_confidence'] = max([float(item[-1]) for item in product["on-shelf_availability"]])  
        
        if product["product_name"] == "Unknown Product":
            product["compliance"] = "unexpected"
        elif data[product["product_name"]]["min"] <= product["count"] <= data[product["product_name"]]["max"]:
            product["compliance"] = "in-stock"
        elif product["count"] < data[product["product_name"]]["min"]:
            product["compliance"] = "low-stock"
        else: product["compliance"] = "overstock"

    return detection_analysis

def summarize_yolo_clip_detections(detection_analysis, llm_analysis): # part 2 of yolo clip detection analysis
    '''
    Summarizes the overall shelf status based on the analysis of detected products, 
    including total products detected, issues identified, and the ratio of unknown products.
    Args:        
        detection_analysis (list): A list of dictionaries containing analysis results for each detected product.
        llm_analysis (dict): A json containing analysis results from LLM reasoning.
    Returns:     detection_summary (dict): A summary of the overall shelf status including total products, issues, and unknown product ratio.
    '''
    # Create a list of misplaced products from LLM results
    misplaced_products = []
    for product in llm_analysis["misplaced_products"]:
        misplaced_products.append(product["product_name"])

    # Analyze the product issues, calculate the ratio of unknown products and add misplaced product details
    product_issues = set()
    unknown_count = 0

    for product in detection_analysis:

        product["compliance"] = [product["compliance"]]
        if product["product_name"] in misplaced_products:
            product["compliance"].append("misplaced")
            product["misplaced_products"] = True
        else:
            product["misplaced_products"] = False

        product_issues.update(product["compliance"])

        if product["product_name"] == "Unknown Product":
            unknown_count += 1
    
    # Generate a detection summary
    detection_summary = {
        "total_products_detected": sum(product["count"] for product in detection_analysis),
        "unknown_ratio": round(unknown_count / len(detection_analysis),2) if detection_analysis else 0,
        "compliance_flag": "ok" if product_issues == {"ok"} else "issues"
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
    
    start_time = time.perf_counter()
    # Perform YOLO + CLIP inference to detect products (using YOLO model) and get their matches (using CLIP embeddings)
    resized_img, detected_products, product_matches = yolo_clip_inference(img)

    # Analyze the detected products
    detection_analysis = analyze_yolo_clip_detections(product_matches)
    
    # Perform LLM reasoning on detected products
    resized_img_shape = resized_img.shape
    llm_analysis = perform_llm_reasoning(detection_analysis, resized_img_shape)
    
    # Summarize the shelf status based on the detections and llm analysis
    detection_summary = summarize_yolo_clip_detections(detection_analysis, llm_analysis)
    
    # Generate a new filename for the uploaded image based on store ID, aisle ID, and timestamp
    new_filename = f"{store_id}_{aisle_id}_{timestamp.strftime('%Y%m%d_%H%M')}_{os.path.splitext(file.filename)[-1]}"
    time1= time.perf_counter()
    print("Total time to run ", time1-start_time) 

    analysis_result = {
        "image_path": new_filename,
        "store_id": store_id,
        "aisle_id": aisle_id,
        "timestamp": timestamp.isoformat(),
        "summary": detection_summary,
        "distribution analysis" : llm_analysis.get("distribution_analysis", {}),
        "compliance actions" : llm_analysis.get("compliance_actions", {}),
        "manager suggestions": llm_analysis.get("manager_suggestions",{}),
        "shelf impact summary": llm_analysis.get("shelf_impact_summary", {}),
        "merchandising evaluation" : llm_analysis.get("merchandising_evaluation", {}),
        "products": detection_analysis
    }
    

    return analysis_result
