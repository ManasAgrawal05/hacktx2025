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
    print("Creating wormhole with dilation enabled...")
    w = wormhole.create(APPID, RELAY_URL, reactor, _enable_dilate=True)

    print("Allocating code...")
    yield w.allocate_code()

    print("Getting code...")
    code = yield w.get_code()
    print(f"Wormhole code: {code}")

    print("Calling dilate()...")
    endpoints = w.dilate()
    print(f"Dilate returned: {endpoints}")

    control_ep, subchannel_client_ep, subchannel_server_ep = endpoints
    print("Dilation established!")

    # Connect using the client endpoint
    print("Opening subchannel connection...")
    port = yield subchannel_client_ep.connect("test-protocol")
    print(f"Connect returned port: {port}")

    protocol = port._protocol
    print("Subchannel connected!")

    # Send test data
    test_message = b"Hello from sender! This is a test message through dilated wormhole."
    protocol.transport.write(test_message)
    print(f"Sent: {test_message}")

    # Wait for response
    yield defer.Deferred().addTimeout(5, reactor)

    protocol.transport.loseConnection()
    yield w.close()
    reactor.stop()


@defer.inlineCallbacks
def receive_side(code):
    print("Creating wormhole with dilation enabled...")
    w = wormhole.create(APPID, RELAY_URL, reactor, _enable_dilate=True)

    print(f"Setting code: {code}")
    w.set_code(code)

    print("Calling dilate()...")
    endpoints = w.dilate()
    print(f"Dilate returned: {endpoints}")

    control_ep, subchannel_client_ep, subchannel_server_ep = endpoints
    print("Dilation established, listening for subchannels...")

    # Listen for incoming subchannel connections
    class ReceiveProtocol:
        def dataReceived(self, data):
            print(f"Received: {data}")
            response = b"Acknowledged! Message received by receiver."
            self.transport.write(response)
            print(f"Sent response: {response}")

        def connectionMade(self):
            print("Subchannel connection established!")

        def connectionLost(self, reason):
            print("Connection closed")

    class ReceiveFactory:
        def buildProtocol(self, addr):
            return ReceiveProtocol()

    print("Starting listener...")
    port = yield subchannel_server_ep.listen("test-protocol", ReceiveFactory())
    print("Listening for 'test-protocol' subchannels...")

    # Keep reactor running
    yield defer.Deferred().addTimeout(10, reactor)

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

    # Errors will propagate with full tracebacks
    d.addErrback(lambda failure: failure.printTraceback())
    reactor.run()


if __name__ == "__main__":
    main()
