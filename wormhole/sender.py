# sender.py
import requests

# Replace this with your tunnel URL
TUNNEL_URL = "https://politicians-influence-crucial-mysimon.trycloudflare.com"

data = "Hello World from Device B!"
resp = requests.post(TUNNEL_URL, data=data)

print(f"Sent: {data}")
print(f"Response: {resp.status_code} {resp.text}")
