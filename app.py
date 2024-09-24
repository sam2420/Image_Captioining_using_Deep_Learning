from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from predicting_caption import predict_caption
import pickle
app = Flask(__name__)

# Load the model and tokenizer
model = load_model('model.h5')
# Assuming you've saved tokenizer during training, load it here as well
# tokenizer = load_tokenizer_function()

# Define a function to preprocess the image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    return image

@app.route('/')
def index():
    return render_template('index.html')

from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

@app.route('/upload', methods=['POST'])
def upload():
    if 'imagefile' in request.files:
        imagefile = request.files['imagefile']
        if imagefile.filename != '':
            image_path = "images/" + imagefile.filename
            imagefile.save(image_path)
            # Preprocess the uploaded image
            image = load_img(image_path, target_size=(224, 224))  # Resize image to (224, 224)
            image = img_to_array(image)  # Convert image to numpy array
            image = preprocess_input(image)  # Preprocess input according to VGG16 requirements
            # Load tokenizer and max_length
            with open('tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            max_length = 35
            # Make prediction using the model
            prediction = predict_caption(model, image, tokenizer, max_length)
            # Redirect to a new page to display the prediction
            return redirect(url_for('prediction', prediction=prediction))
    return redirect(url_for('index'))


@app.route('/prediction/<prediction>')
def prediction(prediction):
    return render_template('prediction.html', prediction=prediction)

@app.route('/home')
def home():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
