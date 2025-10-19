#!/usr/bin/env python3
"""
Test script for bidirectional wormhole communication.

Usage:
  Terminal 1: python wormhole_test.py
  Terminal 2: python wormhole_test.py <code-from-terminal-1>
"""

import sys
import wormhole
from wormhole.cli import public_relay
import threading
import time


def send_messages(endpoint, name):
    """Send test messages through the wormhole."""
    messages = [
        f"Hello from {name}!",
        f"Message 2 from {name}",
        f"Message 3 from {name}",
        "DONE"
    ]

    for msg in messages:
        print(f"[{name}] Sending: {msg}")
        endpoint.send(msg.encode('utf-8'))
        time.sleep(1)


def receive_messages(endpoint, name):
    """Receive messages from the wormhole."""
    while True:
        data = endpoint.receive()
        if data is None:
            print(f"[{name}] Connection closed")
            break

        msg = data.decode('utf-8')
        print(f"[{name}] Received: {msg}")

        if msg == "DONE":
            break


def run_wormhole(code=None):
    """
    Establish dilated wormhole connection.
    If code is None, allocate a new code (sender).
    If code is provided, use that code (receiver).
    """
    # Create wormhole with public relay
    w = wormhole.create(
        appid="example.com/wormhole-test",
        relay_url=public_relay.RENDEZVOUS_RELAY,
        reactor=None  # Use default reactor
    )

    if code is None:
        # Allocate new code - we're the initiator
        code = w.allocate_code()
        print(f"\n=== Your wormhole code: {code} ===\n")
        print("Run this on the other machine:")
        print(f"  python {sys.argv[0]} {code}\n")
        role = "INITIATOR"
    else:
        # Use provided code
        w.set_code(code)
        role = "RECEIVER"

    print(f"[{role}] Establishing connection...")

    # Get versions - required handshake
    versions = w.get_versions()
    print(f"[{role}] Protocol versions: {versions}")

    # Dilate the connection
    print(f"[{role}] Dilating connection...")
    endpoints = w.dilate()

    print(f"[{role}] Connection established! Starting bidirectional test...\n")

    # Start receiver thread
    recv_thread = threading.Thread(
        target=receive_messages,
        args=(endpoints, role)
    )
    recv_thread.start()

    # Give receiver a moment to start
    time.sleep(0.5)

    # Send messages from main thread
    send_messages(endpoints, role)

    # Wait for receiver to finish
    recv_thread.join()

    print(f"\n[{role}] Test complete!")
    w.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Code provided - we're the receiver
        code = sys.argv[1]
        run_wormhole(code)
    else:
        # No code - we're the initiator
        run_wormhole()
