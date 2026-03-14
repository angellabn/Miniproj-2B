import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import run_video_inference, run_image_inference

app = Flask(__name__)
# Enable CORS so the React frontend can communicate with the API
CORS(app)

# Create a temporary directory for uploads if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'temp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict_video():
    """
    Endpoint for analyzing video files.
    EXPECTED PIPELINE:
    Video -> Frame Extraction -> Face Detection (MediaPipe) -> Crop ->
    Resize (224x224) -> 15-frame sequence -> CNN+LSTM -> Prediction
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        # Save file temporarily
        temp_filename = str(uuid.uuid4()) + "_" + file.filename
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(temp_path)
        
        # Run inference pipeline
        result = run_video_inference(temp_path)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during video inference: {e}")
        # Clean up on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500


@app.route('/predict-image', methods=['POST'])
def predict_image():
    """
    Endpoint for analyzing static image files.
    EXPECTED PIPELINE:
    Image -> Face Detection (MediaPipe) -> Crop -> Resize -> Single-frame Model -> Prediction
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        # Read the image file bytes
        image_bytes = file.read()
        
        # Run inference on the image
        result = run_image_inference(image_bytes)
            
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during image inference: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Starting TruthLens backend server on port 5000...")
    app.run(debug=True, port=5000)
