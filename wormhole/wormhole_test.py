#!/usr/bin/env python3
"""
Test script for bidirectional wormhole communication.

Usage:
  Terminal 1: python wormhole_test.py
  Terminal 2: python wormhole_test.py <code-from-terminal-1>
"""

import sys
import wormhole
import threading
import time
from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks

# Public relay server
RELAY_URL = "ws://relay.magic-wormhole.io:4000/v1"


@inlineCallbacks
def run_wormhole(code=None):
    """
    Establish dilated wormhole connection.
    If code is None, allocate a new code (sender).
    If code is provided, use that code (receiver).
    """
    # Create wormhole with public relay
    w = wormhole.create(
        appid="example.com/wormhole-test",
        relay_url=RELAY_URL,
        reactor=reactor
    )

    if code is None:
        # Allocate new code - we're the initiator
        code = yield w.allocate_code()
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
    versions = yield w.get_versions()
    print(f"[{role}] Protocol versions: {versions}")

    # Dilate the connection
    print(f"[{role}] Dilating connection...")
    endpoints = yield w.dilate()

    print(f"[{role}] Connection established! Starting bidirectional test...\n")

    # Send and receive test messages
    for i in range(3):
        msg_out = f"Message {i+1} from {role}"
        print(f"[{role}] Sending: {msg_out}")
        yield endpoints.send(msg_out.encode('utf-8'))

        # Try to receive
        data = yield endpoints.receive()
        if data:
            msg_in = data.decode('utf-8')
            print(f"[{role}] Received: {msg_in}")

    # Send done signal
    yield endpoints.send(b"DONE")

    # Receive final message
    data = yield endpoints.receive()
    if data:
        print(f"[{role}] Received: {data.decode('utf-8')}")

    print(f"\n[{role}] Test complete!")
    yield w.close()
    reactor.stop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Code provided - we're the receiver
        code = sys.argv[1]
        run_wormhole(code)
    else:
        # No code - we're the initiator
        run_wormhole()

    reactor.run()
