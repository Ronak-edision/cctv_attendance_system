import torch
import numpy as np
import chromadb
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image
import cv2
from models import Model
import easyocr


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

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=False, post_process=False, device=torch.device("cpu"))

# Load both fine-tuned FaceNet models
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

transform = transforms.Compose([
  transforms.Resize((160, 160)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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

def get_embedding(image_path, is_outside=False):
    """Generate a FaceNet embedding for an image using adaptive model selection."""
    img = preprocess_image(image_path)
    if img is None:
        return None

    with torch.no_grad():
        if is_outside:
            embedding = facenet_model_92(img)
        else:
            embedding = facenet_model_88(img)

    return embedding.cpu().numpy().flatten().tolist()

def get_stored_embeddings(is_outside=False):
    """Retrieve all stored embeddings from ChromaDB based on whether the footage is outside or inside."""
    collection_to_use = collection_92 if is_outside else collection_88
    stored_data = collection_to_use.get(include=["embeddings", "metadatas"])

    stored_embeddings = {}
    for i, img_id in enumerate(stored_data["ids"]):
        name = img_id.split()[0]  # Extract first word as name
        embedding = np.array(stored_data["embeddings"][i])
        stored_embeddings[name] = torch.tensor(embedding, dtype=torch.float32, device=device)

    return stored_embeddings



def is_face_present_for_numpy(image_array, is_outside=False):

    if is_outside:
        stored_embeddings = get_stored_embeddings(is_outside=True)
    else:
        stored_embeddings = get_stored_embeddings(is_outside=False)

    """Checks if a face is present in an image array using MTCNN."""
    try:
        image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        face = mtcnn(image)
        return face is not None
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

def who(cctv_image_np: np.ndarray, is_outside=False) -> str:
    """Identifies the person in a CCTV image by comparing embeddings with stored embeddings."""
    if is_outside:
        stored_embeddings = get_stored_embeddings(is_outside=True)
    else:
        stored_embeddings = get_stored_embeddings(is_outside=False)
        
    cctv_image = Image.fromarray(cctv_image_np)
    cctv_tensor = transform(cctv_image).unsqueeze(0).to(device)

    with torch.no_grad():
        cctv_emb = facenet_model_92(cctv_tensor) if is_outside else facenet_model_88(cctv_tensor)

    max_score = 0.9
    most_similar_name = "Unknown"
    for name, stored_emb in stored_embeddings.items():
        score = torch.nn.functional.cosine_similarity(cctv_emb, stored_emb.unsqueeze(0)).item()
        if score > max_score:
            max_score = score
            most_similar_name = name

    return most_similar_name

def crop_face_from_numpy(image_np):
    print("Cropping face from image inside function")
    original_image = Image.fromarray(image_np)
    face = mtcnn(original_image)

    if face is not None:
        face_np = face.permute(1, 2, 0).numpy()
        return np.clip(face_np, 0, 255).astype("uint8")
    else:
        print("No face detected!")
        return image_np

def extract_location_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Error: Could not read the first frame from the video."

    reader = easyocr.Reader(['en'])
    results = reader.readtext(frame)

    location_text = results[-1][1] if results else "Location not detected"
    return location_text

