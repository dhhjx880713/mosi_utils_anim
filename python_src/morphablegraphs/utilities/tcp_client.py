import socket


class TCPClient(object):
    """ TCP client that sends and receives a single message
        https://pymotw.com/2/socket/tcp.html
    """
    BUFFER_SIZE = 10485760
    def __init__(self,url, port, buffer_size=BUFFER_SIZE):
        self.address = (url, port)
        self.buffer_size = buffer_size
        self.socket = None#socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def send_message(self,data):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        print "call",self.address, len(data)
        self.socket.connect(self.address)
        self.socket.send(data)
        data = self.socket.recv(self.buffer_size)
        self.socket.close()
        return data