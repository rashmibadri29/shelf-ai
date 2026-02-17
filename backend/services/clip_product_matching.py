import torch
import open_clip
from PIL import Image
import numpy as np
import os
from collections import defaultdict

def load_clip_model():
    ''' 
    Loads the specified CLIP model variant along with its preprocessing transforms and tokenizer,
    and moves the model to the appropriate device (GPU if available, otherwise CPU).
    Returns:     model (torch.nn.Module): The loaded CLIP model.
                 preprocess (callable): The preprocessing transforms associated with the CLIP model variant.
                 tokenizer (callable): The tokenizer associated with the CLIP model variant.
                 device (str): The device (CPU or GPU) on which the model is running.
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu" # Decides whether to run on GPU or CPU

    # Load the CLIP model and preprocessing transforms for the specified variant
    model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    
    # Loads the corresponding tokenizer for the chosen CLIP model variant since previous 
    # tokenizer loading method is deprecated in open_clip v2.0.0 and above
    tokenizer = open_clip.get_tokenizer("ViT-B-32") 

    model = model.to(device)
    model.eval()    # eval() disables dropout etc. (important for inference stability)
    
    return model, preprocess, tokenizer, device

def embed_image(image: Image.Image, model, preprocess, device) -> np.ndarray:
    '''
    Converts a PIL Image to a CLIP embedding vector by applying the necessary preprocessing steps and forwarding it through the CLIP image encoder.
    Args:        image (PIL.Image): The input image to be embedded.
                model: The loaded CLIP model containing the image encoder.
                preprocess: The preprocessing transforms associated with the CLIP model variant.
                device: The device (CPU or GPU) on which the model is running.
    Returns:     image_embedding (np.ndarray): The resulting CLIP embedding vector for the input image.
    '''
    # preprocess: resize + normalize
    image_tensor = preprocess(image).unsqueeze(0).to(device) # unsqueeze(0): add batch dimension [1, C, H, W]

    with torch.no_grad():
        emb = model.encode_image(image_tensor) # Forward pass through CLIP image encoder

    emb = emb / emb.norm(dim=-1, keepdim=True) # L2-normalize → enables cosine similarity via dot product
    
    return emb.cpu().numpy()[0]  # Move to CPU, convert to numpy, remove batch dimension

def embed_text(text: str, model, tokenizer, device) -> np.ndarray:
    '''
    Converts a text string into a CLIP embedding vector by tokenizing the input text and forwarding it through the CLIP text encoder.
    Args:       text (str): The input text string to be embedded.
                model: The loaded CLIP model containing the text encoder.
                tokenizer: The tokenizer associated with the CLIP model variant for converting text to token IDs.
                device: The device (CPU or GPU) on which the model is running.
    Returns:     text_embedding (np.ndarray): The resulting CLIP embedding vector for the input text.
    '''
    text_tokens = tokenizer([text]).to(device) # Convert text prompt into token IDs

    with torch.no_grad():
        emb = model.encode_text(text_tokens) # Forward pass through CLIP text encoder

    emb = emb / emb.norm(dim=-1, keepdim=True) # Same normalization as image embeddings

    return emb.cpu().numpy()[0]

class EmbeddingStore:
    '''
    A simple in-memory store for CLIP embeddings and their associated metadata, with functionality to save/load from disk 
    and perform similarity search. Embeddings and metadata are kept in parallel lists to maintain alignment by index.
    '''
    def __init__(self):
        self.embeddings = []   # List of vectors
        self.metadata = []     # Parallel list of metadata dicts

    def add(self, embedding: np.ndarray, meta: dict): 
        # Add a new embedding and its associated metadata to the store, ensuring that the embedding is stored as a numpy array and that the metadata is aligned by index.
        self.embeddings.append(embedding)
        self.metadata.append(meta)     # Keep embeddings and metadata aligned by index
        
    def as_matrix(self) -> np.ndarray: 
        # Stack all stored embeddings into a single matrix for efficient similarity search.
        return np.vstack(self.embeddings)  # Shape: (num_items, embedding_dim)

    def save_to_disk(self, filepath: str):
        # Save the embeddings and their associated metadata to disk in a compressed .npz format, allowing for later retrieval and use in similarity searches.
        np.savez_compressed(
            filepath,
            embeddings=self.as_matrix(),
            metadata=self.metadata
        )
    
    def load_from_disk(self, filepath: str):
        # Load embeddings and their associated metadata from a compressed .npz file on disk, restoring the in-memory store for use in similarity searches.
        data = np.load(filepath, allow_pickle=True)
        self.embeddings = [emb for emb in data['embeddings']]
        self.metadata = data['metadata'].tolist()
        
def similarity_search(query_emb: np.ndarray, store: EmbeddingStore, top_k: int = 5):
    ''' 
    Performs a similarity search by computing the cosine similarity between a query embedding and all stored embeddings, returning the top-K most similar items along with their metadata.
    Args:        query_emb (np.ndarray): The embedding vector for the query item (image or text).
                store (EmbeddingStore): The in-memory store containing embeddings and their associated metadata.
                top_k (int): The number of top similar items to return based on cosine similarity scores.
    Returns:    results (list): A list of dictionaries containing the top-K most similar items and their metadata.
    '''
    matrix = store.as_matrix()    # Stack all stored embeddings into a single matrix

    scores = matrix @ query_emb  # Dot product = cosine similarity (because vectors are normalized)
    
    top_idx = np.argsort(scores)[-top_k:][::-1]  # Get indices of top-K highest similarity scores

    results = []
    for idx in top_idx:
        # Combine similarity score with product metadata for each of the top-K matches and append to results list
        results.append({
            "score": float(scores[idx]),
            **store.metadata[idx]
        })            

    return results

def aggregate_matches(matches):
    '''
    Aggregates similarity search results by keeping only the highest similarity score for each unique product, and returns the best overall match along with its score.
    Args:        matches (list): A list of dictionaries containing product names and their associated similarity scores from a similarity search.   
    Returns:     best_match (tuple): A tuple containing the best matching product name and its corresponding similarity score.
    '''
    grouped = defaultdict(float)   # product_id → best similarity score
    
    for m in matches:
        grouped[m["product_name"]] = max(
            grouped[m["product_name"]],
            m["score"]
        )   # Keep only the strongest signal per product
    
    best_product, best_score = max(grouped.items(), key=lambda x: x[1])
    return best_product, best_score

def create_and_store_embeddings(filepath: str):
    '''
    Creates CLIP embeddings for all images in the specified file path (either a single image or a directory of images) and stores them in an EmbeddingStore, which is then saved to disk as a compressed .npz file.
    Args:        filepath (str): The path to the image file or directory containing image files.
    Returns:     save_path (str): The path to the saved .npz file containing the embeddings.
    '''
    model, preprocess, tokenizer, device = load_clip_model() # Load CLIP model and related utilities
    store = EmbeddingStore()
    files = []

    print(f"Processing file/directory: {filepath}")
    if os.path.isfile(filepath):
        filename = os.path.basename(filepath)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            files.append(filepath)
    elif os.path.isdir(filepath):
        for root, _, filenames in os.walk(filepath):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    files.append(os.path.join(root, filename))
                else: pass
    else:
        raise ValueError(f"Invalid filepath: {filepath} is neither a file nor a directory.")

    print(f"Found {len(files)} image files to process.")

    for file in files:
        img = Image.open(file).convert('RGB')
        img_emb = embed_image(img, model, preprocess, device)
        text_emb = embed_text(f"{os.path.splitext(os.path.basename(file))[0]}", model, tokenizer, device)
        store.add(img_emb, {"product_name": os.path.splitext(os.path.basename(file))[0], "description": f"Image embedding for {os.path.splitext(os.path.basename(file))[0]}"})
        store.add(text_emb, {"product_name": os.path.splitext(os.path.basename(file))[0], "description": f"Text embedding for {os.path.splitext(os.path.basename(file))[0]}"})

    print(f"Created embeddings for {len(store.embeddings)} images.")

    if os.path.isfile(filepath):
        save_path = os.path.join(os.path.dirname(filepath), os.path.splitext(os.path.basename(filepath))[0] + "_embeddings.npz")
    elif os.path.isdir(filepath):
        save_path = os.path.join(filepath, "embeddings.npz")

    store.save_to_disk(save_path)
    print(f"Embeddings created and saved to {save_path}")

    return save_path

class CLIP_similarity_checker:
    '''
    A class that encapsulates the functionality to perform similarity checks using CLIP embeddings, including loading a pre-built embedding store, 
    converting images to embeddings, and searching for the most similar products based on both image and text queries.
    '''
    def __init__(self, embeddings_db_path: str):
        self.model, self.preprocess, self.tokenizer, self.device = load_clip_model()
        self.store = EmbeddingStore()
        self.store.load_from_disk(embeddings_db_path) # Load previously saved embeddings
    
    def search_embeddings(self, img: Image.Image, text: str = "None", top_k: int = 5):
        ''' Searches for the most similar products in the embedding store based on a given image and optional text query, 
        returning the best matching product and its similarity score. '''
        img_emb = embed_image(img, self.model, self.preprocess, self.device)
        img_matches = similarity_search(img_emb, self.store, top_k)
        
        if text != "None":
            text_emb = embed_text(text, self.model, self.tokenizer, self.device)
            text_matches = similarity_search(text_emb, self.store, top_k)
        else: text_matches = []

        all_matches = img_matches + text_matches

        # for match in all_matches:
        #     print(f"Matched Product: {match['product_name']}, Score: {match['score']:.4f}\n")

        best_product, best_score = aggregate_matches(all_matches)
        
        if best_score < 0.3:  # Threshold for "no good match"
            return "Unknown Product", 0.0
        return best_product, best_score

    def numpy_to_pillow(self, img_array: np.ndarray) -> Image.Image:
        return Image.fromarray((img_array * 255).astype(np.uint8))

    def load_pillow_image(self, img_path: str) -> Image.Image:
        return Image.open(img_path).convert('RGB')

if __name__ == "__main__":

    task = "none" # "none", "create_embeddings" or "similarity_search"

    if task == "create_embeddings": # Create and store embeddings for all images in the specified directory
    
        save_path = create_and_store_embeddings("../data/product_catalog") # Create and store embeddings for all images in the specified directory
    
    elif task == "similarity_search": # Load existing embeddings from disk and perform a similarity search with a new image or text query    
        
        CLIP_checker = CLIP_similarity_checker("../data/clip_embeddings/embeddings.npz") # Initialize the similarity checker with the path to the saved embeddings
        img = CLIP_checker.load_pillow_image("../data/sample_image/Tropicana 100% Juice Apple 10 fl oz.webp") # Load a new image to query against the stored embeddings
        best_product, best_score = CLIP_checker.search_embeddings(img, "Tropicana Apple Juice", top_k=3) # Search for similar products using both image and text queries
        
        print(f"Best Product Overall: {best_product} with Score: {best_score:.4f}\n")
    
    else:
        pass
    
    