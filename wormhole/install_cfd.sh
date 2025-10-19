#!/bin/bash
set -e

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
  FILE="cloudflared-linux-arm64.deb"
elif [ "$ARCH" = "armv7l" ] || [ "$ARCH" = "armhf" ]; then
  FILE="cloudflared-linux-armhf.deb"
else
  echo "Unsupported architecture: $ARCH"
  exit 1
fi

wget "https://github.com/cloudflare/cloudflared/releases/latest/download/$FILE" -O cloudflared.deb
sudo dpkg -i cloudflared.deb || sudo apt -f install -y
cloudflared --version

