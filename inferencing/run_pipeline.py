import socketio

# ===== Configuration =====
TUNNEL_URL = "https://orange-composer-cancer-peripheral.trycloudflare.com"
# ==========================

sio = socketio.Client()


@sio.event
def connect():
    print("✅ Connected to server")


@sio.event
def disconnect():
    print("❌ Disconnected from server")


@sio.on('server_message')
def on_server_message(data):
    print("📨 Server:", data.get('msg'))


def main():
    print("🌐 Connecting to server...")
    sio.connect(TUNNEL_URL, transports=['websocket'])

    try:
        while True:
            input("🔘 Press Enter to send 'take_picture' command...")
            sio.emit('take_picture', {'msg': 'take_picture'})
            print("📤 Sent 'take_picture' command")
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    finally:
        sio.disconnect()


if __name__ == "__main__":
    main()
