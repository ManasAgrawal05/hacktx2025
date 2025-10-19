# 1️⃣ Add Cloudflare GPG key
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null

# 2️⃣ Add the Cloudflare package repository
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/ $(lsb_release -cs) main" \
| sudo tee /etc/apt/sources.list.d/cloudflare-client.list

# 3️⃣ Update and install
sudo apt update
sudo apt install cloudflared -y

