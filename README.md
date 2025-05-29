# Violence Detection with YOLOv8 (SINTESA)

This project is a web-based application built with Flask that uses the YOLOv8 model to detect violence in uploaded video files. The application processes videos locally, annotates frames with bounding boxes where violence is detected (class index 1), and displays both the original and detected videos on a simple web interface. It is optimized for use on a MacBook Air M1 with MPS (Metal Performance Shaders) support, but also supports Windows with CPU-based processing.

## Features

- Upload video files (`.mp4`, `.avi`, `.mov`) up to 100 MB and 60 seconds in duration.
- Real-time violence detection using the pre-trained `yolov8violence_final.pt` model.
- Display of original and annotated (detected) videos side by side.
- Optimized for MacBook Air M1 with GPU acceleration via MPS; Windows uses CPU.
- Automatic cleanup of old uploaded files (older than 1 hour).

## Prerequisites

- **Operating System**: macOS (tested on MacBook Air M1) or Windows 10/11.
- **Python**: Version 3.7 or higher.
- **Dependencies**: See `requirements.txt` for a complete list.
- **Hardware**: At least 8 GB RAM (16 GB recommended for higher resolution videos).

## Installation

### For macOS (MacBook Air M1)

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/violence-detection.git  # Replace with your repository URL
cd violence-detection
```

#### 2. Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Install `ffmpeg` for video conversion:

```bash
brew install ffmpeg
```

#### 4. Prepare the Model

- Place the pre-trained model file `yolov8violence_final.pt` in the project root directory.

#### 5. Run the Application

```bash
python3 app.py
```

### For Windows

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/violence-detection.git  # Replace with your repository URL
cd violence-detection
```

#### 2. Set Up a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Install `ffmpeg` for video conversion:

- Download the latest `ffmpeg` build from [ffmpeg.org](https://ffmpeg.org/download.html) or use a package manager like Chocolatey:
  ```bash
  choco install ffmpeg  # If you have Chocolatey installed
  ```
- Alternatively, download the executable from a trusted source (e.g., [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)), extract it, and add the `ffmpeg.exe` path to your system's environment variables:
  - Right-click on "This PC" > "Properties" > "Advanced system settings" > "Environment Variables".
  - Under "System Variables", find "Path", click "Edit", and add the path to the folder containing `ffmpeg.exe` (e.g., `C:\ffmpeg\bin`).

#### 4. Prepare the Model

- Place the pre-trained model file `yolov8violence_final.pt` in the project root directory.

#### 5. Run the Application

```bash
python app.py
```

Open your browser and go to `http://localhost:5000` to access the application.

## Usage

1. Upload a video file (supported formats: `.mp4`, `.avi`, `.mov`) via the web interface.
2. Click the "Detect" button to process the video.
3. View the "Results" section to see the original video and the detected video with bounding boxes highlighting violence.

## Project Structure

```
violence_detection/
├── app.py                    # Main Flask application
├── yolov8violence_final.pt   # Pre-trained YOLOv8 model
├── static/
│   ├── uploads/              # Folder for uploaded and processed videos
│   └── css/
│       └── style.css         # CSS for styling
├── templates/
│   └── index.html            # HTML template for the web interface
├── requirements.txt          # List of Python dependencies
└── README.md                 # This file
```

## Configuration

- **Upload Limits**: Videos are limited to 100 MB and 60 seconds to ensure smooth processing on local hardware.
- **Device**:
  - macOS: Automatically uses MPS on MacBook M1 if available; otherwise, falls back to CPU.
  - Windows: Uses CPU for processing (CUDA support for NVIDIA GPUs can be added with additional configuration).
- **Frame Processing**: Every second frame is processed to optimize performance on the M1 chip and Windows CPUs.

## Troubleshooting

- **Video Not Displaying**: Ensure `ffmpeg` is installed and the video is converted properly. Check browser console for errors (right-click > Inspect > Console).
- **Model Loading Issues**: Verify `yolov8violence_final.pt` is in the correct directory and compatible with the `ultralytics` version.
- **Performance Issues**: Reduce video resolution or increase frame skip in `app.py` if processing is slow.
- **Access Denied**: If accessing from another device on the same network, ensure the firewall allows port 5000 (on macOS: System Settings > Network > Firewall; on Windows: Windows Defender Firewall > Allow an app through firewall).

## Contributing

Feel free to fork this repository, submit issues, or create pull requests to improve the project. Contributions are welcome!

## License

[Specify your license here, e.g., MIT License] - If you don't have a license yet, consider adding one (e.g., MIT) to define usage terms.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 framework.
- [Flask](https://flask.palletsprojects.com/) for the web framework.
- [OpenCV](https://opencv.org/) for video processing.

## Contact

For questions or support, please open an issue on the repository or contact [your-email@example.com]. # Replace with your contact info
