import zmq

context = zmq.Context()

# create an inproc socket and connect to the endpoint where process A is bound
socket = context.socket(zmq.PAIR)
socket.connect("inproc://my_ipc_endpoint")

# receive a message from process A
received_message = socket.recv()
print(f"Process B received message: {received_message.decode()}")

# send a message back to process A
message = b"Hello from process B"
socket.send(message)
