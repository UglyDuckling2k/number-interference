import os, io, base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# Check model file path
if os.getenv('RUNNING_IN_DOCKER') == 'true':
    model_path = '/model/digit_classifier.pth'
else:
    model_path = './model/digit_classifier.pth'
print(f"Model path: {model_path}")

# Define your CNN model
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  #
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = self.dropout(x) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Flask application
app = Flask(__name__)
CORS(app) 

# Load the trained model
model = DigitClassifier().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Image preprocessing function
def preprocess_image(image_bytes):

    # Convert bytes to a PIL image
    image = Image.open(image_bytes).convert('L')

    # Invert the image colors
    image = ImageOps.invert(image)

    # Calculate the bounding box of the content
    bbox = image.getbbox()
    if bbox:
        # Calculate the center of the image and the bounding box
        image_center = (image.width / 2, image.height / 2)
        bbox_center = ((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2)

        # Calculate the shift needed to center the bounding box
        shift = (image_center[0] - bbox_center[0], image_center[1] - bbox_center[1])

        # Create a new image with the same size and a white background
        new_image = Image.new("L", image.size, "black")
        # Calculate the new bounding box position
        new_bbox = (int(bbox[0] + shift[0]), int(bbox[1] + shift[1]), int(bbox[2] + shift[0]), int(bbox[3] + shift[1]))
        # Paste the content into the new image, centered
        new_image.paste(image.crop(bbox), new_bbox)
        
        image = new_image

    # Resize the image
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert the image to a tensor and normalize it
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    image = transform(image).unsqueeze(0)
    return image

# Define prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Parse JSON data sent with the request
        data = request.get_json()

    # Access the image data from the JSON
    image_data = data["image"]

    # Check if there is data in the request
    if not image_data:
        return jsonify({"error": "No image data"})

    # Decode the base64 string
    try:
        image_bytes = base64.b64decode(image_data)
    except base64.binascii.Error:
        return jsonify({"error": "Invalid base64 encoding"})

    # Preprocess the image
    image = preprocess_image(io.BytesIO(image_bytes))

    # Perform inference
    with torch.no_grad():

        # Ensure the input tensor is on the same device as the model
        image = image.to(device)  

        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    # Return the prediction as JSON response
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
