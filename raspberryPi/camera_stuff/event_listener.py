from flask import Flask
from flask_socketio import SocketIO, emit
import subprocess
import threading
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/')
def index():
    return "Simple take-picture server running."


@socketio.on('connect')
def handle_connect():
    print("✅ Client connected")
    emit('server_message', {'msg': 'Connected to simple receiver'})


@socketio.on('take_picture')
def handle_take_picture(data):
    print("📸 Received take_picture signal:", data)
    emit('server_message', {'msg': 'Taking picture...'})

    def worker():
        try:
            print("▶️ Running raspberryRunner.py...")
            subprocess.run(["~/env/bin/python", "raspberryRunner.py"], shell=True, check=True)
            socketio.emit('server_message', {'msg': 'Picture taken successfully!'})
        except subprocess.CalledProcessError as e:
            print("❌ Error:", e)
            socketio.emit('server_message', {'msg': f"Failed: {e}"})

    # Run in background so server stays responsive
    threading.Thread(target=worker, daemon=True).start()


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8889)
