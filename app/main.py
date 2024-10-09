from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize the FastAPI app
app = FastAPI()

# Load the tokenizer and the model with pre-trained weights
model_path = 'best_bert_model.pth'
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the input format for the API
class TextInput(BaseModel):
    text: str

# Define the available procrastinator types
procrastinator_types = ["Avoider", "Perfectionist", "Indecisive"]

@app.post("/analyze")
async def analyze_text(input: TextInput):
    # Tokenize the input text
    inputs = tokenizer(input.text, return_tensors="pt", padding=True, truncation=True)

    # Predict using the loaded model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    # Map the prediction to the corresponding procrastinator type
    result = {
        "procrastinatorType": procrastinator_types[prediction]
    }

    return result
