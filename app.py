from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure Upload and Output Folders
UPLOAD_FOLDER = "static/uploads"
COMPRESSED_FOLDER = "static/compressed"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB limit

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["COMPRESSED_FOLDER"] = COMPRESSED_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def compress_image(image_path, output_path, quality=85):
    """ Compress an image using K-Means clustering """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Reshape the image to a 2D array of pixels
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Map quality to number of clusters (k)
    k = max(1, min(256, int(quality / 100 * 256)))  # Map quality to 1-256 range

    # K-Means clustering for color quantization
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8 and map labels to center colors
    centers = np.uint8(centers)
    compressed_img = centers[labels.flatten()]
    compressed_img = compressed_img.reshape(img.shape)

    # Convert back to BGR and save
    compressed_img = cv2.cvtColor(compressed_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, compressed_img)
    return os.path.getsize(output_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compress", methods=["POST"])
def compress():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    quality = int(request.form.get("quality", 85))  # Get quality from input

    results = []
    for file in files:
        if file and allowed_file(file.filename):
            if file.content_length > MAX_FILE_SIZE:
                return jsonify({"error": f"{file.filename} exceeds the maximum file size of 5 MB."}), 400

            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            output_path = os.path.join(app.config["COMPRESSED_FOLDER"], filename)

            file.save(input_path)  # Save original image

            original_size = os.path.getsize(input_path)
            compressed_size = compress_image(input_path, output_path, quality)

            results.append({
                "originalUrl": f"/static/uploads/{filename}",
                "compressedUrl": f"/static/compressed/{filename}",
                "originalSize": original_size,
                "compressedSize": compressed_size
            })

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