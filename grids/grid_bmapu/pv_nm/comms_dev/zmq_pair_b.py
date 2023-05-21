import zmq
import time
import random

context = zmq.Context()

# Socket to send messages to process A
sender_socket = context.socket(zmq.PAIR)
sender_socket.connect("tcp://localhost:5556")

# Socket to receive messages from process A
receiver_socket = context.socket(zmq.PAIR)
receiver_socket.connect("tcp://localhost:5555")

while True:
    # Send random float data to process A
    sender_socket.send_json({'p_s':[0.84]*100})

    # Wait for a response from process A
    response = receiver_socket.recv_json()
    #print(f"Received response: {response}")

    #time.sleep(0.001)  # wait for 1 second before sending the next message

# close the socket
sender_socket.close()
receiver_socket.close()

# terminate the context
context.term()