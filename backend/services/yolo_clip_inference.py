from services.yolo_vision import DetectProducts
from services.clip_product_matching import CLIP_similarity_checker
import cv2

def yolo_clip_inference(image_array):
    ''' 
    Performs YOLO object detection to identify products in the input image and then uses CLIP to match each detected product crop 
    against a catalog of product embeddings to determine the most likely product identity along with confidence scores.
    Args:        image_array (np.ndarray): The input image in the form of a NumPy array (BGR format as used by OpenCV).
    Returns:    detected_products (list): A list of detected products with their bounding box coordinates.
                PM_results (list): A list of tuples containing matched product names and their confidence scores for each detected product crop.
    '''

    # Perform YOLO inference to detect products and get bounding boxes
    detect_products = DetectProducts()
    
    image_with_detections, detected_products = detect_products(image_array)
    
    if not detected_products:
        print("No products detected.")
        return {"error": "No products detected."}

    # Get crops of detected products using bounding box coordinates
    crops = detect_products.get_crops()

    # Perform CLIP similarity search for each crop to identify products
    product_matcher = CLIP_similarity_checker("data/clip_embeddings/embeddings.npz")

    # Initialize a list to hold the product matching results
    PM_results = []
    # For each crop, perform a similarity search against the CLIP embedding store to find the best matching product and its confidence score
    for crop in crops:
        pillow_crop = product_matcher.numpy_to_pillow(crop)  # Convert NumPy array to PIL Image
        matched_product, confidence_score = product_matcher.search_embeddings(pillow_crop)
        PM_results.append([matched_product, confidence_score])

    return detected_products, PM_results

def display_results(img, PD_results, PM_results, items_categories=["almond milk", "oat milk", "greek yogurt", "milk",  "yogurt", "cheese", "cream", "butter", "juice", "spread", "kefir", "drink"]):
    ''' 
    Displays the input image with bounding boxes drawn around detected products and annotated with the best matching product name and confidence score from CLIP similarity search results.
    Args:        img (np.ndarray): The original input image in the form of a NumPy array (BGR format as used by OpenCV).
                PD_results (list): A list of detected products with their bounding box coordinates obtained from YOLO inference.
                PM_results (list): A list of tuples containing matched product names and their confidence scores for each detected product crop obtained from CLIP similarity search.
                items_categories (list): A list of product category keywords to look for in the matched product names for annotation purposes.
    Returns:     None (displays the annotated image using OpenCV).
    '''
    for detection, match in zip(PD_results, PM_results):
        if match[0] != "Unknown Product":
            img = cv2.rectangle(img, (detection[0], detection[1]), (detection[2], detection[3]), (0, 255, 0), 2)
            # print(f"Matched Product: {match[0]} with Confidence: {match[1]:.2f}\n")
            found_words = [word for word in match[0].replace(',', ' ').split(' ') if word in items_categories]
            # print(f"Found Words in Matched Product Name: {match[0].replace(',', ' ').split(' ')}\n")
            if not found_words:
                found_words = ["others"]
            img = cv2.putText(img, f"{found_words[0]}", (detection[0]+2,detection[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            img = cv2.putText(img, f"{match[1]:.2f}", (detection[0]+2,detection[1]+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('YOLO + CLIP Inference', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    task = "none" # "none" or "run_test"
    if task == "run_test":
        img = cv2.imread('../data/sample_image/shelf_image_2.jpg')
        detected_products, product_matches = yolo_clip_inference(img)

        display_results(img, detected_products, product_matches)
    else:
        pass