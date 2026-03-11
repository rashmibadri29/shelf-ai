import os
import cv2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from dotenv import load_dotenv

load_dotenv() 

# Pydantic output schema for distribution analysis of products on shelf
class DistributionAnalysis(BaseModel):
    product_category_layout: str = Field(
        description=(
            "High-level description of what types of products "
            "(e.g., milk, juice, dairy, beverages) exist on the shelf "
            "and their approximate spatial placement (e.g., left side, right side, top row, bottom row). "
            "This should summarize category-level organization rather than detailed geometric clustering."
        )
    )

# Pydantic output schema for detected misplaced products
class MisplacedProductFinding(BaseModel):
    product_name: str = Field(
        description="Name of product identified as misplaced"
    )
    count: int
    bounding_boxes: List[List[float]] = Field(
        description="List of bounding boxes (from input) corresponding to the misplaced product"
    )

# Pydantic output schema for merchandise evaluation
class MerchandisingEvaluation(BaseModel):
    brand_blocking: str = Field(
        description="Assessment of whether clear brand blocking is present"
    )
    visibility_of_low_stock: str = Field(
        description="Assessment of visibility of low-stock products"
    )
    layout_consistency: str = Field(
        description="Evaluation of overall layout consistency"
    )

# Pydantic output schema for compliance actions to be taken
class ComplianceActions(BaseModel):
    low_stock: Optional[List[str]] = Field(
        description="Action points addressing all products marked as low-stock"
    )
    overstock: Optional[List[str]] = Field(
        description="Action points addressing all products marked as overstock"
    )
    misplaced: Optional[List[str]] = Field(
        description="Action points addressing all products marked as misplaced"
    )
    unexpected: Optional[List[str]] = Field(
        description="Action points addressing all products marked as unexpected meaning they are unidentified"
    )

# Pydantic output schema for a summary of shelf impact
class ShelfImpactSummary(BaseModel):
    overall_assessment: str = Field(
        description="Overall evaluation of shelf health and urgency of intervention"
    )
    highest_priority_issue: Optional[str] = Field(
        default=None,
        description="Brief description of the most critical issue detected"
    )

# Pydantic output schema for analysis resport
class SpatialAnalysisReport(BaseModel):
    distribution_analysis: DistributionAnalysis
    misplaced_products: List[MisplacedProductFinding]
    merchandising_evaluation: MerchandisingEvaluation
    compliance_actions: ComplianceActions
    shelf_impact_summary: ShelfImpactSummary
    manager_suggestions: List[str] = Field(
        min_items=3,
        max_items=10
    )

# Return Instructions/Prompt for the LLM
def get_instructions():
    instructions = """
    You are a retail shelf merchandising analyst.

    You are given structured shelf data including:
    - Products grouped in low-stock_products, overstock_products, misplaced_products, unexpected_products, in_stock_products
    - Each category has a list of products dictionaries containing:
        - Product name
        - Counts
        - List of [[Bounding box], [average position], confidence] for every item belonging to that product. Bounding box format = [x1, y1, x2, y2]; average position format = [x_center_norm, y_center_norm]. 
    - Bounding boxes use pixel coordinates. Average position is the normalized center point of the product on the image. 
        - Origin (0,0) is top-left of the image.
        - x increases to the right.
        - y increases downward.
        
            
    IMPORTANT RULES:
    - Do NOT modify counts.
    - Do NOT generate new bounding boxes.
    - Use only bounding boxes provided as input.
    - Perform reasoning only.

    Perform the following analysis:

    1. Distribution Analysis:
    Provide a high-level summary of:
        - What types of product categories exist (e.g., milk, juice, dairy, beverages).
        - Where these categories are positioned on the shelf (left/right/top/bottom/center).

    2. Misplaced Product Findings:
        - Identify EVERY product that appears misplaced based on:
            • Category grouping inconsistencies
            • Large spatial distance from similar products
            • Layout irregularities
        - For each misplaced product return:
            • product_name
            • count of misplaced items
            • Include bounding boxes that indicate the misplaced items
        - If none exist, return an empty list.

    3. Merchandising Evaluation:
        - Assess brand blocking.
        - Assess visibility of low-stock items.
        - Assess overall layout consistency.

    4. Compliance Actions:
        - You are given explicit product lists under:
            • low_stock_products
            • overstock_products
            • misplaced_products
            • unexpected_products
            • in_stock_products
        - For EACH product in the input array of low_stock, overstock and unexpected; and EACH misplaced product from misplaced product finding analysis: 
            - Generate an actionable recommendation 
            - Every object in the input lists (low_stock, overstock and unexpected) and misplaced product finding MUST appear in the output.
            - Summarize entries and group actions wherever possible
        - Ignore the products under in_stock_products
        - If a category list is empty, return an empty list
        
    5. Provide 3–7 high-level general suggestions for improving shelf layout and sales effectiveness.

    Return your response strictly following the provided JSON schema and make it compact.
    Avoid long sentences.
    Each action point should be a short phrase.
    Do not add extra fields.
    Do NOT change object counts.
    If a section has no findings, return an empty list where applicable. """
        
    return instructions

def get_modified_detection_analysis(image_shape, detection_analysis):
    '''
    Modifies detection analysis by grouping products based on compliance 
    Args:
        image_shape (tuple): Shape of resized image in the format (height, width, channel)
        detection_analysis (list): A list of dictionaries containing analysis results for each detected product.
    Returns: modified_detection_analysis (dict): A dict of products grouped by their compliance metric
    '''
    (height,width,ch) = image_shape
    # Creates backbone
    modified_detection_analysis = {"low_stock_products": [], "overstock_products": [], "misplaced_products":[], "unexpected_products": [], "in_stock_products": []}

    # Loops through each product from detection analysis and updates it to new dictionary, adds average position metric
    for Dict in detection_analysis:
        for i,[bbox,conf] in enumerate(Dict['on-shelf_availability']):
            average_position = [round((bbox[0]+bbox[2])/(2*width),2),round((bbox[1]+bbox[3])/(2*height),2)]
            Dict['on-shelf_availability'][i] = [bbox, average_position, conf]

        if Dict["compliance"]=='low-stock':
            modified_detection_analysis["low_stock_products"].append({"product_name": Dict["product_name"], "count": Dict["count"], 'on-shelf_availability': Dict['on-shelf_availability']})
        elif Dict["compliance"]=='overstock':
            modified_detection_analysis["overstock_products"].append({"product_name": Dict["product_name"], "count": Dict["count"], 'on-shelf_availability': Dict['on-shelf_availability']})
        elif Dict["compliance"]=='misplaced':
            modified_detection_analysis["misplaced_products"].append({"product_name": Dict["product_name"], "count": Dict["count"], 'on-shelf_availability': Dict['on-shelf_availability']})
        elif Dict["compliance"]=='unexpected':
            modified_detection_analysis["unexpected_products"].append({"product_name": Dict["product_name"], "count": Dict["count"], 'on-shelf_availability': Dict['on-shelf_availability']})
        elif Dict["compliance"]=='in-stock':
            modified_detection_analysis["in_stock_products"].append({"product_name": Dict["product_name"], "count": Dict["count"], 'on-shelf_availability': Dict['on-shelf_availability']})

         
    return modified_detection_analysis 

def perform_llm_reasoning(detection_analysis, image_shape):
    # Gets LLM prompt
    instructions = get_instructions()

    # Gets modified detection analysis with products grouped by compliance
    modified_detection_analysis = get_modified_detection_analysis(image_shape, detection_analysis)
    
    # Initializes LLM using LangChain
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=os.environ['GOOGLE_API_KEY'],
        temperature=0,
        top_p = 1,
        max_output_tokens = 7000
    )

    # Binds the Pydantic model to enforce the JSON schema
    structured_llm = llm.with_structured_output(schema=SpatialAnalysisReport.model_json_schema(), method="json_schema")

    # Formats the contents as a list of strings inside a HumanMessage
    # LangChain typically joins these into a single prompt string
    contents = [
        f"detection_analysis = {modified_detection_analysis}", 
        instructions
    ]

    # Invoke the chain
    response = structured_llm.invoke([
        HumanMessage(content="\n".join(contents))
    ])

    return response

