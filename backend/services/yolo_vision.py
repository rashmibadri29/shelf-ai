from typing import List, Dict
from ultralytics import YOLO
import numpy as np
import cv2


class DetectProducts:
    ''' 
    A class that utilizes a pre-trained YOLO model to detect products in an input image, extract their bounding boxes, 
    and provide functionality to retrieve cropped images of detected products for further analysis.
    '''
    def __init__(self, model_weights:str = "yolov8n.pt") -> None:
        self.model = YOLO(model_weights)  # Load the pre-trained YOLO model

    def __call__(self, image_array: np.ndarray) -> (np.array, List[Dict]):
        ''' Takes an input image as a NumPy array, performs object detection using the YOLO model, and returns the original image 
        along with a list of detected products, each containing bounding box coordinates, confidence scores, and class labels.
        Args:        image_array (np.ndarray): The input image in the form of a NumPy array (BGR format as used by OpenCV).
        Returns:     results (tuple): A tuple containing the original image array and a list of dictionaries, where each dictionary represents 
        a detected product with its bounding box coordinates, confidence score, and class label.
        '''
        self.image_array = image_array
        # Perform YOLOv8 inference on the input image
        detections = self.model(self.image_array, conf=0.01)  # Adjust confidence threshold as needed    
        
        # Initialize list to hold detection results
        self.results = []
        
        # Custom OpenCV BBox labels and confidence
        for box in detections[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            label = f"{self.model.names[int(box.cls[0])]}"
            confidence = f"{box.conf[0]:.2f}"
            self.results.append([x1, y1, x2, y2, box.conf[0].item(), self.model.names[int(box.cls[0])]])
            
        return self.image_array, self.results

    def display_image(self):
        ''' Displays the image with detected products using OpenCV.'''
        # Display the image using OpenCV
        cv2.imshow('Input image', self.image_array)
        
        # Wait for a key press and then close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_crops(self) -> List[np.ndarray]:
        ''' Extracts and returns cropped images of detected products based on the bounding box coordinates obtained from YOLO inference.
        Returns:     crops (List[np.ndarray]): A list of NumPy arrays, where each array is a cropped image of a detected product extracted from the original input image using the bounding box coordinates.
        '''
        # Extract crops of detected products using bounding box coordinates
        if not self.results:
            print("No detections to crop.")
            return []
        crops = []
        for det in self.results:
            x1, y1, x2, y2 = det[:4]
            crop = self.image_array[y1:y2, x1:x2]
            crops.append(crop)
        return crops

    def image_with_detections(self) -> np.ndarray:
        ''' Returns the image with detected products overlaid with bounding boxes, labels, and confidence scores.
        Returns:     img_with_detections (np.ndarray): The input image with bounding boxes, labels, and confidence scores overlaid.
        '''
        # Custom OpenCV BBox Plotting with labels and confidence
        for box in self.results:
            x1, y1, x2, y2 = box[:4]
            label = f"{box[5]}"
            confidence = f"{box[4]:.2f}"
            img_with_detections = cv2.rectangle(self.image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img_with_detections = cv2.putText(img_with_detections, label, (x1+2, y1 +15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            img_with_detections = cv2.putText(img_with_detections, confidence, (x1+2, y1 +35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img_with_detections

if __name__ == "__main__":
    task = "none" # "none" or "run_test"
    if task == "run_test":
        detect_products = DetectProducts()  # Initialize the YOLO detection class

        img = cv2.imread('../data/sample_image/shelf_image_2.jpg')
        image_array, detected_products = detect_products(img)  # Example usage with a dummy image

        if not detected_products:
            print("No products detected.")
            exit()

        clss = detected_products[0][5] # Print first detected product details
        confidence = detected_products[0][4] # Print first detected product confidence
        print(f"Detected Product: {clss} with Confidence: {confidence:.2f}")
    else:
        pass