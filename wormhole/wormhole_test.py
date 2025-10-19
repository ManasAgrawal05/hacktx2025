#!/usr/bin/env python3
"""
Test script for Magic Wormhole dilation.
Usage:
    # On machine 1 (sender):
    python wormhole_dilation_test.py send

    # On machine 2 (receiver):
    python wormhole_dilation_test.py receive <code>
"""

import sys
from twisted.internet import reactor, defer
from wormhole import wormhole

APPID = "example.com/wormhole-dilation-test"
RELAY_URL = "ws://relay.magic-wormhole.io:4000/v1"


@defer.inlineCallbacks
def send_side():
    w = wormhole.create(APPID, RELAY_URL, reactor)
    yield w.allocate_code()
    code = yield w.get_code()
    print(f"Wormhole code: {code}")
    print("Waiting for receiver to connect...")

    # Establish dilation
    endpoints = yield w.dilate()
    ep = endpoints[0]
    print("Dilation established, connecting...")

    # Get the socket
    socket = yield ep.connect()
    print("Socket connected!")

    # Send test data
    test_message = b"Hello from sender! This is a test message through dilated wormhole."
    socket.write(test_message)
    print(f"Sent: {test_message}")

    # Wait for response
    response = yield socket.receive(1024)
    print(f"Received response: {response}")

    socket.close()
    yield w.close()
    reactor.stop()


@defer.inlineCallbacks
def receive_side(code):
    w = wormhole.create(APPID, RELAY_URL, reactor)
    yield w.input_code(code)
    print(f"Using code: {code}")
    print("Connecting to sender...")

    # Establish dilation
    endpoints = yield w.dilate()
    ep = endpoints[0]
    print("Dilation established, listening...")

    # Get the socket
    socket = yield ep.connect()
    print("Socket connected!")

    # Receive data
    data = yield socket.receive(1024)
    print(f"Received: {data}")

    # Send response
    response = b"Acknowledged! Message received by receiver."
    socket.write(response)
    print(f"Sent response: {response}")

    socket.close()
    yield w.close()
    reactor.stop()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "send":
        d = send_side()
    elif mode == "receive":
        if len(sys.argv) < 3:
            print("Error: receive mode requires a wormhole code")
            print(__doc__)
            sys.exit(1)
        code = sys.argv[2]
        d = receive_side(code)
    else:
        print(f"Error: unknown mode '{mode}'")
        print(__doc__)
        sys.exit(1)

    d.addErrback(lambda failure: failure.printTraceback())
    reactor.run()


if __name__ == "__main__":
    main()
