import zmq

context = zmq.Context()

# create an inproc socket and bind it to a known endpoint
socket = context.socket(zmq.PAIR)
socket.bind("inproc://my_ipc_endpoint")

# send a message to process B
message = b"Hello from process A"
socket.send(message)

# receive a message from process B
received_message = socket.recv()
print(f"Process A received message: {received_message.decode()}")
