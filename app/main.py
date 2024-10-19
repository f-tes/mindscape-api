from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import gdown  # Make sure to include this in your dependencies

# Initialize the FastAPI app
app = FastAPI()

# Google Drive file ID for the model
drive_url = 'https://drive.google.com/uc?id=1l3DqN5_CkBz-IlUVCt8GvW4OPWQWW4xG'
model_path = 'best_bert_model.pth'

# Function to download the model from Google Drive
def download_model():
    if not os.path.exists(model_path):
        print("Downloading model weights from Google Drive...")
        gdown.download(drive_url, model_path, quiet=False)
    else:
        print("Model weights already exist.")

# Download the model weights
download_model()

# Load the tokenizer and the model with pre-trained weights
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the input format for the API
class TextInput(BaseModel):
    text: str

# Define the available procrastinator types
procrastinator_types = [
    "Perfectionist",   # 1
    "Anxious",         # 2
    "Distracted",      # 3
    "Decisional",      # 4
    "Overwhelmed",     # 5
    "Thrillseeker",    # 6
    "Multitasker",     # 7
    "Theorist"         # 8
]

@app.post("/analyze")
async def analyze_text(input: TextInput):
    # Tokenize the input text
    inputs = tokenizer(input.text, return_tensors="pt", padding=True, truncation=True)

    # Predict using the loaded model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()  # Get the predicted class (0-7)

    # Map the prediction to the corresponding procrastinator type
    result = {
        "procrastinatorType": procrastinator_types[prediction]
    }

    return result
