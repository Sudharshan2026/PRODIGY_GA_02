import os
import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_from_directory, send_file

app = Flask(__name__)

# Load the Keras CV model
model = keras_cv.models.StableDiffusion(img_height=512, img_width=512)

def plot_images(images, save_path):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.savefig(save_path)
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        batch_size = int(request.form.get('batch_size', 1))
        images = model.text_to_image(prompt, batch_size=batch_size)
        image_path = os.path.join('static', 'generated_image.png')
        plot_images(images, image_path)
        return render_template('index.html', image_path=image_path)
    return render_template('index.html', image_path=None)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/download_image')
def download_image():
    return send_file('static/generated_image.png', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
