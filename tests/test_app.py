import os
import pytest
from flask import Flask
from app import app, allowed_file, clean_uploads
import tempfile
import time
from io import BytesIO
import shutil
from unittest.mock import patch, MagicMock
from werkzeug.datastructures import FileStorage

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
    with app.test_client() as client:
        yield client
    # Cleanup after test
    shutil.rmtree(app.config['UPLOAD_FOLDER'])

def test_allowed_file():
    # Test allowed file extensions
    assert allowed_file('video.mp4') == True
    assert allowed_file('video.avi') == True
    assert allowed_file('video.mov') == True
    
    # Test disallowed file extensions
    assert allowed_file('image.jpg') == False
    assert allowed_file('document.pdf') == False
    assert allowed_file('noextension') == False

def test_clean_uploads():
    # Create temporary directory with files
    temp_dir = tempfile.mkdtemp()
    
    # Create a file that's "old" (modify access time to be > 1 hour ago)
    old_file = os.path.join(temp_dir, 'old_file.mp4')
    with open(old_file, 'w') as f:
        f.write('test')
    old_time = time.time() - 3700  # 1 hour + 100 seconds ago
    os.utime(old_file, (old_time, old_time))
    
    # Create a new file
    new_file = os.path.join(temp_dir, 'new_file.mp4')
    with open(new_file, 'w') as f:
        f.write('test')
    
    # Mock app.config and glob to use our temp directory
    with patch('app.app.config', {'UPLOAD_FOLDER': temp_dir}):
        with patch('app.glob.glob', return_value=[old_file, new_file]):
            clean_uploads()
    
    # Check that old file is removed but new file remains
    assert not os.path.exists(old_file)
    assert os.path.exists(new_file)
    
    # Cleanup
    shutil.rmtree(temp_dir)

def test_index_get(client):
    response = client.get('/')
    assert response.status_code == 200

@patch('app.model')
def test_index_post_no_file(mock_model, client):
    response = client.post('/')
    assert response.status_code == 200
    assert b'No file uploaded' in response.data

@patch('app.model')
def test_index_post_empty_filename(mock_model, client):
    response = client.post('/', data={
        'file': (FileStorage(filename=''), 'file', '')
    })
    assert response.status_code == 200
    assert b'No file selected' in response.data

@patch('app.model')
def test_index_post_invalid_extension(mock_model, client):
    response = client.post('/', data={
        'file': (BytesIO(b'test data'), 'test.txt')
    })
    assert response.status_code == 200
    # This should fail silently because it never gets to an error message
    # as the file is not in allowed_extensions

@patch('app.convert_video_for_browser', return_value=True)
@patch('app.cv2.VideoCapture')
@patch('app.cv2.VideoWriter')
@patch('app.model')
def test_index_post_valid_file(mock_model, mock_writer, mock_capture, mock_convert, client):
    # This test would need extensive mocking of cv2, os, and other components
    # Here's a simplified version that doesn't test the full video processing
    
    # Mock the video capture and writer
    mock_capture_instance = MagicMock()
    mock_capture_instance.isOpened.return_value = True
    mock_capture_instance.get.side_effect = lambda x: 640 if x == 3 else 480 if x == 4 else 30.0
    mock_capture_instance.read.side_effect = [(True, MagicMock()), (False, None)]
    mock_capture.return_value = mock_capture_instance
    
    mock_writer_instance = MagicMock()
    mock_writer_instance.isOpened.return_value = True
    mock_writer.return_value = mock_writer_instance
    
    # Mock the model results
    mock_results = MagicMock()
    mock_results.__len__.return_value = 1
    mock_results.__getitem__.return_value.plot.return_value = MagicMock()
    mock_model.return_value = [mock_results]
    
    # Mock os.path functions
    with patch('os.path.exists', return_value=True), \
         patch('os.path.getsize', return_value=1024), \
         patch('os.remove'):
        
        # Create a test video file
        video_content = b'fake video content'
        response = client.post('/', data={
            'file': (BytesIO(video_content), 'test.mp4')
        }, content_type='multipart/form-data')
        
        assert response.status_code == 200
        # Check that processing was successful
        assert b'result_test.mp4' in response.data