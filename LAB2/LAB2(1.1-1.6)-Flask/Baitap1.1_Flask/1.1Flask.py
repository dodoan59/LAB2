# -*- coding: utf-8 -*-
"""1.1-1.6(Flask).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wY1SByALkyO0vtdpUZpQNsHCM4ZsZCrq
"""

from PIL import Image
import numpy as np
from flask import Flask, render_template, send_file

app = Flask(__name__)

@app.route('/')
def home():
    # Open image
    img = Image.open('bird.png').convert('L')

    # Convert image to ndarray
    im_1 = np.asarray(img)

    # Inversion operation
    im_2 = 255 - im_1

    # Create new image from ndarray
    new_img = Image.fromarray(im_2)
    
    # Save new image temporarily
    temp_image_path = 'temp_image.png'
    new_img.save(temp_image_path)

    # Rendering HTML template with the image
    return render_template('index_1.1.html', image_path=temp_image_path)

@app.route('/image')
def get_image():
    # Send the saved image file
    return send_file('temp_image.png', mimetype='image/png')

if __name__ == '__main__':
    app.run()
