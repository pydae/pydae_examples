import zmq
import json
import collections.abc
import time

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

data = {}
while True:
    message = socket.recv()

    data_client = json.loads(message)
    
    t_0 = time.perf_counter_ns()
    update(data,data_client)
    t_1 = time.perf_counter_ns()
    print((t_1-t_0)/1000)
    
            
    response = json.dumps(data) 
    socket.send(response.encode())


