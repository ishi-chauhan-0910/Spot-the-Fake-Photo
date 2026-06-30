import os
import time
import base64
import tempfile
from flask import Flask, render_template, request, jsonify
from predict import predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        # The image data is a data URL: 'data:image/jpeg;base64,...'
        image_data_url = data['image']
        header, encoded = image_data_url.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        
        # Save to a temporary file
        fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        with os.fdopen(fd, 'wb') as f:
            f.write(image_bytes)
            
        # Run prediction and measure latency
        start_time = time.perf_counter()
        score = predict(temp_path)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000.0
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'score': score,
            'latency_ms': latency_ms
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("Starting Spot the Fake Photo Live Demo...")
    print("Open http://127.0.0.1:5000 in your browser to try it out!")
    # Disable debug mode to prevent reloading issues if model loads
    app.run(host='127.0.0.1', port=5000, debug=False)
