import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
from io import BytesIO
from pathlib import Path

# Определяем абсолютные пути к файлам
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fast_bird_classifier_mobilenet.pth"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

class BirdClassifier:
    def __init__(self, model_path, class_names_path):
        self.device = torch.device("cpu")
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        
        self.model = models.efficientnet_b0(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    async def predict(self, image_bytes: bytes) -> dict:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = self.class_names[predicted_idx.item()]
        
        return {
            "bird_species": predicted_class,
            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            "confidence": f"{confidence.item()*100:.2f}%" 
        }

# Используем новые абсолютные пути для создания экземпляра
model_handler = BirdClassifier(
    model_path=MODEL_PATH,
    class_names_path=CLASS_NAMES_PATH
)



