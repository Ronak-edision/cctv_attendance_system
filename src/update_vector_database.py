import os
import numpy as np
import torch
import chromadb
from torchvision import transforms
from PIL import Image
from models import Model

# Check and set the device
if torch.backends.mps.is_available():  
    device = torch.device("mps")
elif torch.cuda.is_available():  
    device = torch.device("cuda")
else:  # Default to CPU
    device = torch.device("cpu")

# Paths to ChromaDB storage for each model
CHROMA_DB_PATH_92 = "data/embeddings/model_92"
CHROMA_DB_PATH_88 = "data/embeddings/model_88"

# Initialize ChromaDB clients with persistent storage for each model
chroma_client_92 = chromadb.PersistentClient(path=CHROMA_DB_PATH_92)
chroma_client_88 = chromadb.PersistentClient(path=CHROMA_DB_PATH_88)

# Create or get existing collections for each model
collection_92 = chroma_client_92.get_or_create_collection(name="face_embeddings_model_92")
collection_88 = chroma_client_88.get_or_create_collection(name="face_embeddings_model_88")

# Load the fine-tuned FaceNet models
state_dict_92 = torch.load("models/facenet/fine_tuned/best_model_92.pth", map_location=device)
state_dict_88 = torch.load("models/facenet/fine_tuned/model_88_67.pth", map_location=device)

# Initialize models
facenet_model_92 = Model()
facenet_model_88 = Model()

# Load state dictionaries
facenet_model_92.load_state_dict(state_dict_92)
facenet_model_88.load_state_dict(state_dict_88)

# Move models to device and set to evaluation mode
facenet_model_92.to(device).eval()
facenet_model_88.to(device).eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the tensor
])

def preprocess_image(image_path):
    """Load and preprocess an image for FaceNet."""
    try:
        with Image.open(image_path).convert('RGB') as img:
            img = transform(img).unsqueeze(0).to(device)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
def get_embedding(image_path, model_type="92"):
    """Generate a FaceNet embedding for an image."""
    img = preprocess_image(image_path)
    if img is None:
        return None
    with torch.no_grad():
        if model_type == "92":
            embedding = facenet_model_92(img)
        else:
            embedding = facenet_model_88(img)
    return embedding.cpu().numpy().flatten().tolist()  # Convert to list for ChromaDB

# Store embedding function
def store_embeddings(image_folder, source_type="bizos", model_type="92"):
    """
    Extract embeddings and store them in ChromaDB with source metadata, avoiding duplicates.
    """

    # Select the appropriate collection based on the model type
    if model_type == "92":
        collection = collection_92
    else:
        collection = collection_88

    # Get existing entries in the collection to check for duplicates
    existing_entries = collection.get()
    existing_ids = existing_entries['ids'] if existing_entries else []

    # Only delete if there are existing embeddings to delete
    if existing_ids:
        collection.delete(ids=existing_ids)

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        emb = get_embedding(img_path, model_type=model_type)
        
        if emb is not None:
            print(f"Embedding for {img_name} from {model_type}: {emb[:5]}...")  # Print first 5 values
            print(f"Shape of embedding for {img_name}: {np.shape(emb)}")

            # Add the embedding to the appropriate collection
            collection.add(
                ids=[img_name],
                embeddings=[emb],
                metadatas=[{"source": source_type, "filename": img_name}]
            )
            print(f"Stored {img_name} from {source_type} in {model_type} database.")
        else:
            print(f"Failed to extract embedding for {img_name}, skipping.")


# Store Bizos (anchor) images as permanent for model_92
bizos_folder = "data/raw/bizos_export"
store_embeddings(bizos_folder, source_type="bizos", model_type="92")

# Store CCTV images for model_88 (outside footage)
cctv_folder = "data/raw/bizos_export"
store_embeddings(cctv_folder, source_type="bizos", model_type="88")