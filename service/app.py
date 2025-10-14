from flask import Flask, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from flask_cors import CORS
import requests


app = Flask(__name__)
CORS(app) 

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@app.route('/caption', methods=['POST'])
def caption_image_api():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    image = Image.open(file).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return jsonify({"caption": caption})


if __name__ == "__main__":
    app.run(debug=True)
