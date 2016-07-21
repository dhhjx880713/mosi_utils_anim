import socket


class TCPClient(object):
    """ TCP client that sends and receives a single message
        https://pymotw.com/2/socket/tcp.html
    """
    BUFFER_SIZE = 5000
    def __init__(self,url, port, buffer_size=BUFFER_SIZE):
        self.address = (url, port)
        self.buffer_size = buffer_size

    def send_message(self,data):
        print "call",self.address
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(self.address)
        s.send(data)
        data = s.recv(self.buffer_size)
        s.close()
        return data