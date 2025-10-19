import socketio
import subprocess
import threading
import eventlet
import eventlet.wsgi

# Create Socket.IO server
sio = socketio.Server(async_mode='eventlet')
app = socketio.WSGIApp(sio)


@sio.event
def connect(sid, environ):
    print(f"✅ Client connected: {sid}")
    sio.emit('server_message', {'msg': 'Connected to Raspberry Pi'}, to=sid)


@sio.event
def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")


@sio.on('take_picture')
def handle_take_picture(sid, data):
    print(f"📸 Received take_picture signal: {data}")
    sio.emit('server_message', {'msg': 'Taking picture...'}, to=sid)

    def worker():
        try:
            print("▶️ Running raspberryRunner.py...")
            subprocess.run(
                ["~/env/bin/python", "raspberryRunner.py"],
                shell=True,
                check=True
            )
            sio.emit('server_message', {'msg': 'Picture taken successfully!'}, to=sid)
        except subprocess.CalledProcessError as e:
            print("❌ Error running script:", e)
            sio.emit('server_message', {'msg': f'Error: {e}'}, to=sid)

    threading.Thread(target=worker, daemon=True).start()


if __name__ == '__main__':
    port = 8888
    print(f"🚀 Raspberry Pi receiver listening on port {port} ...")
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', port)), app)
