from flask import Flask, request, render_template, url_for, send_from_directory
from ultralytics import YOLO
import os
import cv2
import time
import glob
import torch
from werkzeug.utils import secure_filename
from moviepy import VideoFileClip

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Batas 100 MB untuk video

# Pastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model YOLOv8
model = YOLO('yolov8violence_final.pt')

# Optimasi untuk M1 (MPS) atau CPU
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model.to(device)
print(f"Using device: {device}")

# Cek ekstensi file yang diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Bersihkan file lama (>1 jam)
def clean_uploads():
    for f in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')):
        if os.path.getmtime(f) < time.time() - 3600:
            os.remove(f)

# Konversi video untuk kompatibilitas browser
def convert_video_for_browser(input_path, output_path):
    try:
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec='libx264', audio=False, fps=clip.fps)
        clip.close()
        return True
    except Exception as e:
        print(f"Conversion error: {e}")
        return False

# Route untuk melayani file statis dari uploads
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='video/mp4')

# Route untuk halaman utama dan deteksi video
@app.route('/', methods=['GET', 'POST'])
def index():
    clean_uploads()

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No file selected')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Proses video
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                os.remove(filepath)
                return render_template('index.html', error='Error opening video')

            # Dapatkan resolusi dan frame rate
            width = int(cap.get(3))  # Lebar
            height = int(cap.get(4))  # Tinggi
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default 30 FPS jika tidak terdeteksi

            output_filename = f"result_{filename}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            temp_output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{output_filename}")

            # Gunakan codec sementara untuk OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                cap.release()
                os.remove(filepath)
                return render_template('index.html', error='Failed to initialize video writer')

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                # Proses setiap frame ke-2 untuk kecepatan
                if frame_count % 2 != 0:
                    out.write(frame)  # Tulis frame asli jika dilewati
                    continue
                results = model(frame, classes=[1], device=device)
                if results and len(results) > 0:
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                else:
                    out.write(frame)  # Tulis frame asli jika tidak ada deteksi

            cap.release()
            out.release()

            # Konversi video untuk kompatibilitas browser
            if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                if convert_video_for_browser(temp_output_path, output_path):
                    os.remove(temp_output_path)  # Hapus file sementara
                    return render_template('index.html',
                                        original_video=f'/static/uploads/{filename}',
                                        result_video=f'/static/uploads/{output_filename}?t={int(time.time())}')
                else:
                    os.remove(temp_output_path)
                    os.remove(filepath)
                    return render_template('index.html', error='Failed to convert video for browser')
            else:
                os.remove(filepath)
                return render_template('index.html', error='Failed to generate detected video')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)