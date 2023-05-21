import socket
import json
import collections.abc

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

UDP_IP = "0.0.0.0"
UDP_PORT = 5568
bufferSize = 1024

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

data = {}
while True:
    
    message, address = sock.recvfrom(bufferSize)
    data_client = json.loads(message)
    
    update(data,data_client)
            
    data_server = json.dumps(data) 
    sock.sendto(str.encode(data_server), address)
    #print(data)