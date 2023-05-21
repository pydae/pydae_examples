import zmq
import time
import threading

def publisher():
    context = zmq.Context()
    publisher_socket = context.socket(zmq.PUB)
    publisher_socket.bind("tcp://*:5563")
    topic = "news"
    while True:
        news = "The time is " + time.ctime(time.time())
        publisher_socket.send_string("%s %s" % (topic, news))
        time.sleep(1)

def subscriber():
    context = zmq.Context()
    subscriber_socket = context.socket(zmq.SUB)
    subscriber_socket.connect("tcp://localhost:5563")
    topic_filter = "news"
    subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, topic_filter)
    while True:
        message = subscriber_socket.recv_string()
        topic, news = message.split(" ", 1)
        print("Received %s: %s" % (topic, news))
        # Respond to the message
        response_topic = "response"
        response_message = "Received %s at %s" % (topic, time.ctime(time.time()))
        publisher_socket.send_string("%s %s" % (response_topic, response_message))

if __name__ == "__main__":
    context = zmq.Context()
    publisher_socket = context.socket(zmq.PUB)
    publisher_socket.bind("tcp://*:5559")
    publisher_thread = threading.Thread(target=publisher)
    subscriber_thread = threading.Thread(target=subscriber)
    publisher_thread.start()
    subscriber_thread.start()
