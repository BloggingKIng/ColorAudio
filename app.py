import matplotlib
matplotlib.use('Agg') 
import colorsys
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
from docx import Document
import PyPDF2
import pdfplumber
app = Flask(__name__)

def read_docx(file):
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def read_pdf(file):
    full_text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            full_text.append(page.extract_text())
    return '\n'.join(full_text)

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

def generate_color_palette():
    chars = string.digits + string.ascii_lowercase + string.ascii_uppercase + '!?., '
    palette = {}
    
    num_colors = len(chars)
    for i, char in enumerate(chars):
        hue = i**2 / num_colors   
        lightness = 0.3 + (i % 2) * 0.4 
        saturation = 0.8 + (i % 3) * 0.6 
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        rgb = tuple(int(255 * x) for x in rgb)
        palette[char] = rgb
    palette['A'] = (310,210,200)
    return palette

def char_to_color(char, palette):
    return palette.get(char, (100,100,100)) 

def color_to_char(color, palette):
    for char, col in palette.items():
        if col == color:
            return char
    return '?'

def string_to_color_pattern(input_string, palette, cell_width=200, cell_height=200):
    length = len(input_string)
    width = length * cell_width
    height = cell_height + cell_height // 2  
    image = Image.new("RGB", (width, height), (255, 255, 255))  
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=50)


    color_code = []
    for i, char in enumerate(input_string):
        color = char_to_color(char, palette)
        color_code.append(color)
        top_left = (i * cell_width, 0)
        bottom_right = ((i + 1) * cell_width, cell_height)
        draw.rectangle([top_left, bottom_right], fill=color)
        text_width, text_height = 20,20
        text_x = top_left[0] + (cell_width - text_width) / 2
        text_y = cell_height + (cell_height // 2 - text_height) / 2
        draw.text((text_x, text_y), char, fill=(0, 0, 0), font=font, stroke_width=1)
    
    return image, color_code

def color_code_to_string(color_code, palette):
    return ''.join(color_to_char(tuple(color), palette) for color in color_code)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
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
        input_string = ""
        if 'text' in request.form and request.form['text']:
            input_string = request.form['text']
        elif 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            filename = file.filename.lower()
            if filename.endswith('.txt'):
                input_string = file.read().decode('utf-8')
            elif filename.endswith('.docx'):
                input_string = read_docx(file)
            elif filename.endswith('.pdf'):
                input_string = read_pdf(file)
            else:
                return render_template('text_to_color.html', error="Unsupported file format")
        else:
            return render_template('text_to_color.html', error="No input provided")
        
        palette = generate_color_palette()
        image, color_code = string_to_color_pattern(input_string, palette)
        
        image_chunks = split_image_into_chunks(image, chunk_size=200)  

        return render_template('text_to_color_representation.html', image_chunks=image_chunks, color_code=color_code)

    return render_template('text_to_color.html')

@app.route('/color-to-text', methods=['GET', 'POST'])
def color_to_text():
    if request.method == 'POST':
        if 'color_code' in request.form and request.form['color_code']:
            color_code_input = request.form['color_code']
            print(color_code_input.strip())
            color_code = color_code_input.strip().strip('][').split('),')
            cz = []
            for x in color_code:
                z = x.strip().strip('(').strip(')')
                z = z.split(',')
                u,v,w = z
                z = (int(u),int(v),int(w))
                cz.append(tuple(z))
            print(cz)
            print(color_code)
            palette = generate_color_palette()
            original_text = color_code_to_string(cz, palette)
            return render_template('color_to_text.html', original_text=original_text, color_code=color_code_input)
        else:
            return render_template('color_to_text.html', error="No input provided")
    return render_template('color_to_text.html')

if __name__ == '__main__':
    app.run(debug=True)

