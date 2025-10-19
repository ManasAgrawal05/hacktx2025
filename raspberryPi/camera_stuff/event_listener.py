import socketio
import threading
import subprocess

# Create Socket.IO server (threading mode = no async or SSL issues)
sio = socketio.Server(async_mode='threading')
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
    print(f"📸 Received take_picture signal from {sid}: {data}")
    sio.emit('server_message', {'msg': 'Taking picture...'}, to=sid)

    def worker():
        try:
            print("▶️ Running raspberryRunner.py...")
            subprocess.run(
                ["~/env/bin/python", "raspberryRunner.py"],
                shell=True,
                check=True
            )
            print("✅ Picture taken successfully!")
            sio.emit('server_message', {'msg': 'Picture taken successfully!'}, to=sid)
        except subprocess.CalledProcessError as e:
            print("❌ Error running script:", e)
            sio.emit('server_message', {'msg': f'Error: {e}'}, to=sid)

    # Run command in background so server remains responsive
    threading.Thread(target=worker, daemon=True).start()


if __name__ == '__main__':
    import wsgiref.simple_server

    port = 8888
    print(f"🚀 Raspberry Pi receiver listening on port {port} ...")
    server = wsgiref.simple_server.make_server('0.0.0.0', port, app)
    server.serve_forever()
