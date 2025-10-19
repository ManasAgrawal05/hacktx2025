from aiohttp import web
import asyncio
import socketio
import subprocess

# Create async Socket.IO server
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins="*")

app = web.Application()
sio.attach(app)


@sio.event
async def connect(sid, environ):
    print(f"✅ Client connected: {sid}")
    await sio.emit('server_message', {'msg': 'Connected to Raspberry Pi'}, to=sid)


@sio.event
async def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")


@sio.on('take_picture')
async def handle_take_picture(sid, data):
    print(f"📸 Received take_picture signal: {data}")
    await sio.emit('server_message', {'msg': 'Taking picture...'}, to=sid)

    # Run raspberryRunner.py asynchronously so server stays responsive
    loop = asyncio.get_event_loop()

    def run_script():
        try:
            print("▶️ Running raspberryRunner.py...")
            subprocess.run(["$HOME/env/bin/python",
                            "$HOME/hacktx2025/raspberryPi/camera_stuff/raspberryRunner.py"],
                           shell=True,
                           check=True)
            print("✅ Picture taken successfully!")
            asyncio.run_coroutine_threadsafe(
                sio.emit('server_message', {'msg': 'Picture taken successfully!'}, to=sid),
                loop
            )
        except subprocess.CalledProcessError as e:
            print("❌ Error running script:", e)
            asyncio.run_coroutine_threadsafe(
                sio.emit('server_message', {'msg': f'Error: {e}'}, to=sid),
                loop
            )
    loop.run_in_executor(None, run_script)

if __name__ == '__main__':
    print("🚀 Raspberry Pi receiver listening on port 8898 ...")
    web.run_app(app, host='0.0.0.0', port=8889)
