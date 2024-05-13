import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from flask import Flask, render_template, request, redirect, url_for, send_file
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    if file:
        y, sr = librosa.load(file)
        D = np.abs(librosa.stft(y))
        D_db = librosa.amplitude_to_db(D, ref=np.max)
        
        colors = [(0, 'black'), (0.25, 'blue'), (0.5, 'green'), (0.75, 'yellow'), (1, 'red')]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log', cmap=cmap, ax=ax)
        ax.axis('off')  # Remove axes
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0)  # Adjust padding
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('color_representation.html', plot_url=plot_url)

@app.route('/color_representation')
def color_representation():
    plot_url = request.args.get('plot_url')
    return render_template('color_representation.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
