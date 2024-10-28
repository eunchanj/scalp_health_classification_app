from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Load the EfficientNet model
MODEL_PATH = 'model.pt'  # Update with your actual .pt file path
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image as expected by EfficientNet
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization used in EfficientNet
])

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return render_template('index2.html', prediction='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index2.html', prediction='No selected file')
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        image = Image.open(filepath).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(input_tensor)
            if output.numel() == 1:
                prediction = torch.sigmoid(output).item()
            else:
                prediction = torch.sigmoid(output)[0][1].item()  # Use the first element if output has multiple elements
            print(prediction)
            prediction_class = '위험할 수도,,' if prediction > 0.8 else '건강해요:)'
        
        # Return the result
        return render_template('index2.html', prediction=f'Prediction: {prediction_class}')
    return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=True)
