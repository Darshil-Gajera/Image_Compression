from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
import threading

app = Flask(__name__)

# Configure Upload and Output Folders
UPLOAD_FOLDER = "static/uploads"
COMPRESSED_FOLDER = "static/compressed"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE = 16 * 1024 * 1024  # Increased to 16 MB
MAX_IMAGE_DIMENSION = 1024  # Maximum dimension for large images

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["COMPRESSED_FOLDER"] = COMPRESSED_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)

# Thread-local storage for handling concurrent requests
thread_local = threading.local()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_if_needed(img):
    """Resize image if it's too large while maintaining aspect ratio"""
    height, width = img.shape[:2]
    if max(height, width) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return img

def compress_image(image_path, output_path, quality=85):
    """Compress an image using optimized K-Means clustering"""
    # Read image and resize if needed
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image")
    
    img = resize_if_needed(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape and prepare for clustering
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Optimize k based on image size and quality
    k = max(2, min(128, int(quality / 100 * 128)))  # Reduced max clusters for speed
    
    # Optimize K-means parameters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)  # Reduced iterations
    attempts = 1  # Reduced attempts for speed
    
    # Perform K-means clustering
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    
    # Convert back to image format using vectorized operations
    centers = np.uint8(centers)
    compressed_img = centers[labels.flatten()].reshape(img.shape)
    
    # Convert back to BGR and save with optimal compression
    compressed_img = cv2.cvtColor(compressed_img, cv2.COLOR_RGB2BGR)
    
    # Use optimal compression parameters
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, min(quality, 95)]  # Limit maximum quality
    cv2.imwrite(output_path, compressed_img, encode_params)
    
    return os.path.getsize(output_path)

def process_single_file(file, quality, app_config):
    """Process a single file for parallel execution"""
    filename = secure_filename(file.filename)
    input_path = os.path.join(app_config["UPLOAD_FOLDER"], filename)
    output_path = os.path.join(app_config["COMPRESSED_FOLDER"], filename)
    
    file.save(input_path)
    original_size = os.path.getsize(input_path)
    compressed_size = compress_image(input_path, output_path, quality)
    
    return {
        "originalUrl": f"/static/uploads/{filename}",
        "compressedUrl": f"/static/compressed/{filename}",
        "originalSize": original_size,
        "compressedSize": compressed_size
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compress", methods=["POST"])
def compress():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    quality = int(request.form.get("quality", 85))

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file in files:
            if file and allowed_file(file.filename):
                futures.append(
                    executor.submit(process_single_file, file, quality, app.config)
                )
        
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing file: {str(e)}")

    return jsonify(results)

@app.route("/reset", methods=["POST"])
def reset():
    """Delete all uploaded and compressed images"""
    for folder in [UPLOAD_FOLDER, COMPRESSED_FOLDER]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
    return jsonify({"message": "All images have been reset."})

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True)