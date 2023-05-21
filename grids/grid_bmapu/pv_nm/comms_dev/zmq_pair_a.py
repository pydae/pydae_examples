import zmq
import time
import random
import numpy as np

context = zmq.Context()

# Socket to send messages to process B
sender_socket = context.socket(zmq.PAIR)
sender_socket.bind("tcp://*:5555")

# Socket to receive messages from process B
receiver_socket = context.socket(zmq.PAIR)
receiver_socket.bind("tcp://*:5556")

while True:
    # Send random float data to process B
    t_0 = time.perf_counter_ns()
    sender_socket.send_json({'p_s_ppc':[0.85]*100})

    # Wait for a response from process B
    t_1 = time.perf_counter_ns()
    response = receiver_socket.recv_json()

    t_2 = time.perf_counter_ns()
    if 'p_s' in response:
        p_s = np.array(response['p_s'])  
    t_3 = time.perf_counter_ns()

    print(f"t_1-t_0: {(t_1-t_0)/1e6:4.3f}, t_2-t_1: {(t_2-t_1)/1e6:4.3f}, t_3-t_2: {(t_3-t_2)/1e6:4.3f} ms")

    time.sleep(0.05)  # wait for 1 second before sending the next message
