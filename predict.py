import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

disease_classes = ["blight", "grey_leaf_spot", "healthy", "rust"]
pest_classes = ["aphids", "corn_rootworm", "fall_army_worm", "stalk_borer"]

print("Loading CLIP model for maize check...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("CLIP loaded!\n")

def is_maize(image):
    maize_texts = [
        "a photo of a maize leaf with long narrow shape and parallel veins",
        "a photo of a corn plant leaf that is long and narrow",
        "a photo of maize crop in a farm field",
        "a photo of corn crop showing disease or pest damage",
    ]
    non_maize_texts = [
        "a photo of a round or oval shaped leaf",
        "a photo of a person or human",
        "a photo of an animal",
        "a photo of food or vegetables",
        "a photo of a building or object",
        "a photo of a spinach leaf",
        "a photo of a banana leaf",
        "a photo of a mango leaf",
        "a photo of a random plant that is not maize",
    ]

    all_texts = maize_texts + non_maize_texts

    inputs = clip_processor(
        text=all_texts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    maize_score = probs[:len(maize_texts)].sum().item()
    non_maize_score = probs[len(maize_texts):].sum().item()

    print(f"Maize score: {maize_score:.2f} | Non-maize score: {non_maize_score:.2f}")
    return maize_score > non_maize_score and maize_score > 0.35

import gdown
import os

os.makedirs("models", exist_ok=True)

if not os.path.exists("models/maize_check_model.pth"):
    gdown.download("https://drive.google.com/uc?id=1z6HDhi2jB59cDmcncut1rqxl0or9eCNg", "models/maize_check_model.pth", quiet=False)

if not os.path.exists("models/maize_disease_model.pth"):
    gdown.download("https://drive.google.com/uc?id=1k727yTOBtOLosOzDUhRBfrTq0HYeB-Ej", "models/maize_disease_model.pth", quiet=False)

if not os.path.exists("models/maize_pest_model.pth"):
    gdown.download("https://drive.google.com/uc?id=15T6JUZ8JZ_Ydx3Q09W8lQVatnH9MNx4B", "models/maize_pest_model.pth", quiet=False)
    
def load_models():
    disease_model = models.resnet18(weights=None)
    disease_model.fc = nn.Linear(disease_model.fc.in_features, len(disease_classes) )
    disease_model.load_state_dict(torch.load(
        "models/maize_disease_model.pth", map_location=device))
    disease_model.to(device)
    disease_model.eval()

    pest_model = models.resnet18(weights=None)
    pest_model.fc = nn.Linear(pest_model.fc.in_features, len(pest_classes))
    pest_model.load_state_dict(torch.load(
        "models/maize_pest_model.pth", map_location=device))
    pest_model.to(device)
    pest_model.eval()

    return None, disease_model, pest_model

def predict_image(image, maize_model, disease_model, pest_model):
    if not is_maize(image):
        return "not_maize", 0.0, "none"

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        disease_output = disease_model(img)
        disease_probs = torch.softmax(disease_output, dim=1)
        disease_conf, disease_pred = torch.max(disease_probs, 1)

        pest_output = pest_model(img)
        pest_probs = torch.softmax(pest_output, dim=1)
        pest_conf, pest_pred = torch.max(pest_probs, 1)

    if disease_conf.item() >= pest_conf.item():
        return (
            disease_classes[disease_pred.item()],
            round(disease_conf.item() * 100, 1),
            "disease"
        )
    else:
        return (
            pest_classes[pest_pred.item()],
            round(pest_conf.item() * 100, 1),
            "pest"
        )
