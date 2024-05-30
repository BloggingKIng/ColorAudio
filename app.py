import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from flask import Flask, render_template, request
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import string
import math

app = Flask(__name__)

# Function to split image into chunks
def split_image_into_chunks(image, chunk_size):
    width, height = image.size
    chunks = []
    for i in range(0, width, chunk_size):
        chunk = image.crop((i, 0, min(i + chunk_size, width), height))
        buf = io.BytesIO()
        chunk.save(buf, format='PNG')
        buf.seek(0)
        chunks.append(base64.b64encode(buf.getvalue()).decode())
    return chunks

# Function to generate color palette for characters
def generate_color_palette():
    chars = string.digits + string.ascii_uppercase
    palette = {}
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (192, 192, 192), (128, 128, 128), (128, 0, 0),
        (128, 128, 0), (0, 128, 0), (128, 0, 128),
        (0, 128, 128), (0, 0, 128), (255, 165, 0),
        (255, 20, 147), (0, 191, 255), (50, 205, 50),
        (186, 85, 211), (188, 143, 143), (210, 105, 30),
        (105, 105, 105), (255, 69, 0), (128, 128, 0),
        (70, 130, 180), (154, 205, 50), (255, 140, 0),
        (153, 50, 204), (72, 209, 204), (255, 105, 180),
        (147, 112, 219), (123, 104, 238), (0, 250, 154),
        (244, 164, 96), (127, 255, 0), (240, 230, 140)
    ]

    for i, char in enumerate(chars):
        palette[char] = colors[i % len(colors)]
    return palette

# Convert character to color
def char_to_color(char, palette):
    return palette.get(char.upper(), (255, 255, 255))  # Default to white

# Convert string to color pattern image
def string_to_color_pattern(input_string, palette, cell_width=200, cell_height=200):
    length = len(input_string)
    width = length * cell_width
    height = cell_height + cell_height // 2  # Additional space for text
    image = Image.new("RGB", (width, height), (255, 255, 255))  # Initialize with white background
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=cell_height // 4)
    except IOError:
        font = ImageFont.load_default()
    for i, char in enumerate(input_string):
        color = char_to_color(char, palette)
        top_left = (i * cell_width, 0)
        bottom_right = ((i + 1) * cell_width, cell_height)
        draw.rectangle([top_left, bottom_right], fill=color)
        text_width, text_height = 20,20
        text_x = top_left[0] + (cell_width - text_width) / 2
        text_y = cell_height + (cell_height // 2 - text_height) / 2
        draw.text((text_x, text_y), char, fill=(0, 0, 0), font=font, stroke_width=1)
    return image


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    if file:
        y, sr = librosa.load(file)
        D = np.abs(librosa.stft(y))
        D_db = librosa.amplitude_to_db(D, ref=np.max)
        
        num_colors = 256
        colors = [(0, 'black')]
        for i in range(1, num_colors):
            frequency = i / num_colors * (sr / 2)
            hue = frequency / (sr / 2)
            colors.append((i / (num_colors - 1), plt.cm.hsv(hue)[:3]))
        
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log', cmap=cmap, ax=ax)
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Decibels')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode()

        return render_template('color_representation.html', plot_url=plot_url)

@app.route('/text-to-color', methods=['GET', 'POST'])
def text_to_color():
    if request.method == 'POST':
        if 'text' in request.form and request.form['text']:
            input_string = request.form['text']
        elif 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            input_string = file.read().decode('utf-8')
        else:
            return render_template('text_to_color.html', error="No input provided")
        
        palette = generate_color_palette()
        image = string_to_color_pattern(input_string, palette)
        image_chunks = split_image_into_chunks(image, chunk_size=200)  # Adjust
                # Split the image into chunks
        image_chunks = split_image_into_chunks(image, chunk_size=200)  # Adjust chunk size as needed

        # Pass the image chunks to the template
        return render_template('text_to_color_representation.html', image_chunks=image_chunks)

    # Render the text_to_color.html template for GET requests
    return render_template('text_to_color.html')

if __name__ == '__main__':
    app.run(debug=True)
