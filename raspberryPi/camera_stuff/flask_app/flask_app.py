from flask import Flask, send_file, jsonify
from io import BytesIO
from noise_visualizer import NoiseVisualizer

app = Flask(__name__)
visualizer = NoiseVisualizer()

@app.route("/clean", methods=["GET"])
def get_clean_image():
    try:
        image = visualizer.get_clean_image()
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dirty", methods=["GET"])
def get_dirty_image():
    try:
        image = visualizer.get_dirty_image()
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "NoiseVisualizer Flask API",
        "endpoints": {
            "/clean": "Get clean image",
            "/dirty": "Get dirty image with overlay"
        }
    })
