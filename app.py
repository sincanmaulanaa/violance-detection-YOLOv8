from flask import Flask, request, render_template, url_for, send_from_directory
from ultralytics import YOLO
import os
import cv2
import time
import glob
import torch
import requests
from werkzeug.utils import secure_filename
from moviepy import VideoFileClip
import telegram
import asyncio

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Batas 100 MB untuk video

# Konfigurasi Telegram
TELEGRAM_BOT_TOKEN = "7638807782:AAEvQJmNZCWhOSmoaBpUZ4LOqymdfMCzCLc"
TELEGRAM_CHAT_ID = "1185853665"

# Inisialisasi bot Telegram dengan konfigurasi connection pool
bot = telegram.Bot(
    token=TELEGRAM_BOT_TOKEN,
    request=telegram.request.HTTPXRequest(
        connection_pool_size=50,  # Tingkatkan ukuran pool
        pool_timeout=30.0         # Timeout 30 detik
    )
)

# Buat event loop global
loop = asyncio.get_event_loop()

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

# Add this function after the import statements

def parse_filename_metadata(filename):
    """
    Parse metadata from filename with format ROOM_DATE_TIME.mp4
    Example: D404_11-06-25_11-00.mp4 -> Room D404, 11 June 2025, 11:00
    """
    try:
        # Remove file extension
        name_without_ext = filename.rsplit('.', 1)[0]
        
        # Split by underscore
        parts = name_without_ext.split('_')
        
        if len(parts) != 3:
            return None
            
        room, date_str, time_str = parts
        
        # Parse date (format: DD-MM-YY)
        day, month, year = date_str.split('-')
        # Add 20 prefix to year if it's 2 digits
        if len(year) == 2:
            year = f"20{year}"
            
        # Format date in a more readable way
        months = ['January', 'February', 'March', 'April', 'May', 'June', 
                 'July', 'August', 'September', 'October', 'November', 'December']
        formatted_date = f"{int(day)} {months[int(month)-1]} {year}"
        
        # Parse time (format: HH-MM)
        hour, minute = time_str.split('-')
        formatted_time = f"{hour}:{minute} WIB"
        
        return {
            'room': room,
            'date': formatted_date,
            'time': formatted_time
        }
    except Exception as e:
        print(f"Error parsing filename metadata: {e}")
        return None

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

# Kirim notifikasi ke Telegram
def send_telegram_notification_sync(message):
    try:
        print(f"Attempting to send to chat_id: {TELEGRAM_CHAT_ID}")
        api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(api_url, data=payload)
        if response.status_code == 200:
            print(f"Telegram notification sent: {message}")
            print(f"Telegram API response: {response.json()}")
            return True
        else:
            print(f"Telegram API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Failed to send Telegram notification: {e}")
        return False
    
def send_telegram_photo(photo_path, caption):
    try:
        with open(photo_path, 'rb') as photo:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {'photo': photo}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}
            response = requests.post(url, files=files, data=data)
            return response.status_code == 200
    except Exception as e:
        print(f"Error sending photo: {e}")
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
            metadata = parse_filename_metadata(filename)

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
            violence_detected = False  # Flag untuk mendeteksi kekerasan
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                # Proses setiap frame ke-2 untuk kecepatan
                if frame_count % 2 != 0:
                    out.write(frame)
                    continue
                results = model(frame, classes=[1], device=device)
                if results and len(results) > 0 and len(results[0].boxes) > 0:
                    # Kekerasan terdeteksi
                    violence_detected = True
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                else:
                    out.write(frame)

            cap.release()
            out.release()


            # Kirim notifikasi ke Telegram jika kekerasan terdeteksi
            if violence_detected:
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "violence_frame.jpg"), annotated_frame)
                photo_path = os.path.join(app.config['UPLOAD_FOLDER'], "violence_frame.jpg")

                # Extract metadata from filename if available
                metadata_str = ""
                metadata = parse_filename_metadata(filename)
                if metadata:
                    metadata_str = f"\nRuangan: {metadata['room']}\nTanggal: {metadata['date']}\nWaktu: {metadata['time']}"
                
                message = f"⚠️ Tindak kekerasan terjadi pada video: {filename}{metadata_str}\nDiproses pada: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                notification_sent = send_telegram_notification_sync(message)
                photo_caption = "Cuplikan tindak kekerasan pada video"
                if metadata:
                    photo_caption += f" - Ruangan: {metadata['room']}"
                photo_sent = send_telegram_photo(photo_path, photo_caption)
                if not notification_sent:
                    print("Warning: Could not send Telegram notification")

            # Konversi video untuk kompatibilitas browser
            if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                if convert_video_for_browser(temp_output_path, output_path):
                    os.remove(temp_output_path)  # Hapus file sementara
                    return render_template('index.html',
                                        original_video=f'/static/uploads/{filename}',
                                        result_video=f'/static/uploads/{output_filename}?t={int(time.time())}', metadata=metadata)
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