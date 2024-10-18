import torch
import torch.nn as nn
from transformers import AutoModelForVideoClassification
from transformers import  ViTForImageClassification, ViTImageProcessor
import logging
import joblib

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize device (CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load ViT model and processor
model_save_path = "./10epochs_model"
model = ViTForImageClassification.from_pretrained(model_save_path).to(device)
processor = ViTImageProcessor.from_pretrained(model_save_path)

# Define class names
class_names = ['ع', 'ا', 'ے', 'ب', 'ث', 'چ', 'ی', 'د', 'ڈ', 'ض', 'ف', 'گ', 'غ', 'ژ', 'ہ', 'ء', 'ح', 'ج', 'ک', 'خ', 'ل', 'م', 'ن', 'پ', 'ق', 'ر', 'ڑ', 'س', 'ش', 'ص', 'ت', 'ٹ', 'ط', 'و', 'ذ', 'ز', 'ظ']

class LandmarkViViT(nn.Module):
    def __init__(self, pretrained_model, input_dim, d_model, num_classes):
        super(LandmarkViViT, self).__init__()
        self.fc1 = nn.Linear(input_dim, d_model)  # First linear layer to map inputs
        self.layer_norm = nn.LayerNorm(d_model)  # Layer normalization to stabilize training
        self.fc2 = nn.Linear(d_model, num_classes)  # Output layer to predict classes
        self.vivit_model = pretrained_model  # Use ViViT model

    def forward(self, x):
        x = self.fc1(x)  # First linear transformation
        x = self.layer_norm(x)  # Apply layer normalization
        transformer_output = self.vivit_model.vivit.encoder(x, return_dict=True).last_hidden_state
        output = self.fc2(transformer_output.mean(dim=1))  # Average pooling followed by output layer
        return output

# Function to load model and scaler
def load_model_and_scaler(model_path, scaler_path, num_classes):
    # Instantiate the Transformer model
    input_dim = 1662
    d_model = 768
    
    model_name = "google/vivit-b-16x2-kinetics400"
    pretrained_vivit_model = AutoModelForVideoClassification.from_pretrained(model_name).to(device)

    # Initialize the model
    landmark_model = LandmarkViViT(pretrained_model=pretrained_vivit_model, input_dim=input_dim, d_model=d_model, num_classes=num_classes).to(device)

    # Load the saved state dictionary into the model
    try:
        landmark_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        logging.info(f"Model loaded from {model_path}")
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")

    # Ensure the model is in evaluation mode for inference
    landmark_model.eval()

    # Load the scaler
    try:
        scaler = joblib.load(scaler_path)
        logging.info(f"Scaler loaded from {scaler_path}")
    except Exception as e:
        logging.error(f"Error loading scaler from {scaler_path}: {e}")
        scaler = None  # Set scaler to None if loading fails

    return landmark_model, scaler

# Define categories
categories = {
    'greeting': {
        'model_path': 'greeting.pth',
        'scaler_path': 'greeting.pkl',
        'actions': [
            "السلام وعلیکم",
            "خدا حافظ",
            "ایک اچھا دن گزاریں",
            "بعد میں ملتے ہیں",
            "خوش آمدید"
        ]
    },
    'daily_routine': {
        'model_path': 'everyday.pth',
        'scaler_path': 'everyday.pkl',
        'actions': [
            "ایمبولینس کو کال کریں",
            "کیا میں آپ کا حکم لے سکتا ہوں؟",
            "میں بیمار ہوں",
            "میں نے پوری رات مطالعہ کیا",
            "چلو ایک ریستوراں میں چلو"
        ]
    },
    'question': {
        'model_path': 'question.pth',
        'scaler_path': 'question.pkl',
        'actions': [
            "کیا تم بھوکے ہو؟",
            "آپ کیسے ہیں؟",
            "اس کی کیا قیمت ہے؟",
            "میں نہیں سمجھا",
            "آپ کا ٹیلیفون نمبر کیا ہے؟"
        ]
    }
}

# Initialize models and scalers for all categories
def initialize_categories():
    initialized_categories = {}
    for category_name, data in categories.items():
        # Load model and scaler for each category
        model, scaler = load_model_and_scaler(
            model_path=data['model_path'],
            scaler_path=data['scaler_path'],
            num_classes=len(data['actions'])
        )
        initialized_categories[category_name] = {
            'model': model,
            'scaler': scaler,
            'actions': data['actions']
        }
    return initialized_categories

# Initialize and store in a variable
initialized_categories = initialize_categories()
