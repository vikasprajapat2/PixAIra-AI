from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from datetime import datetime
from models import text_to_image, image_to_text

app = Flask(__name__)
UPLOAD_FOLDER = 'static/generated/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text_to_image', methods=['POST'])
def text2image():
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({'error': 'NO Prompt provided'}), 400
    
    print(f"Received text-to-image request for prompt: {prompt}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        image = text_to_image(prompt)
        image.save(filepath)
        return jsonify({"image_url": f"/static/generated/{filename}"})
    except Exception as e:
        print(f"Error generating image: {e}")
        return jsonify({'error': str(e)}), 500
  
@app.route("/image-to-text", methods=['POST'])
def image2text():
    if "image" not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({'error': 'No image selected'}), 400

    temp_path = os.path.join("static/generated", "temp_upload.jpg")
    file.save(temp_path)

    try:
        caption = image_to_text(temp_path)
        return jsonify({'caption': caption})
    except Exception as e:
        print(f"Error generating caption: {e}")
        return jsonify({'error': str(e)}), 500

if __name__== "__main__":
    app.run(debug=True)