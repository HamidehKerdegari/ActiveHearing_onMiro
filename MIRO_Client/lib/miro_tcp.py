#!/usr/bin/env python

#######################################################################################################################
import thread, time, os


def cl_black(msge): return '\033[30m'+msge+'\033[0m'
def cl_red(msge): return '\033[31m'+msge+'\033[0m'
def cl_green(msge): return '\033[32m'+msge+'\033[0m'
def cl_orange(msge): return '\033[33m'+msge+'\033[0m'
def cl_blue(msge): return '\033[34m'+msge+'\033[0m'
def cl_purple(msge): return '\033[35m'+msge+'\033[0m'
def cl_cyan(msge): return '\033[36m'+msge+'\033[0m'
def cl_lightgrey(msge): return '\033[37m'+msge+'\033[0m'
def cl_darkgrey(msge): return '\033[90m'+msge+'\033[0m'
def cl_lightred(msge): return '\033[91m'+msge+'\033[0m'
def cl_lightgreen(msge): return '\033[92m'+msge+'\033[0m'
def cl_yellow(msge): return '\033[93m'+msge+'\033[0m'
def cl_lightblue(msge): return '\033[94m'+msge+'\033[0m'
def cl_pink(msge): return '\033[95m'+msge+'\033[0m'
def cl_lightcyan(msge): return '\033[96m'+msge+'\033[0m'

#######################################################################################################################
#   Class: Kuka iiwa TCP communication    #####################
class server:
    #   M: __init__ ===========================
    def __init__(self, ip, port):
        self.BUFFER_SIZE = 1024
        self.isconnected = False
        self.JointPosition = ([None,None,None,None,None,None,None],None)
        self.isready = False
        

        try:
            # Starting connection thread
            thread.start_new_thread( self.socket, (ip, port, ) )
        except:
            print cl_red('Error: ') + "Unable to start connection thread"
    #   ~M: __init__ ==========================

    #   M: Stop connection ====================
    def close(self):
        self.isconnected = False
    #   ~M: Stop connection ===================

    #   M: Connection socket ==================
    def socket(self, ip, port):
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Bind the socket to the port
        server_address = (ip, port)

        print cl_cyan('Starting up on:'), 'IP:', ip, 'Port:', port
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(server_address)
        except:
            print cl_red('Error: ') + "Connection for KUKA cannot assign requested address:", ip, port
            os._exit(-1)


        # Listen for incoming connections
        sock.listen(1)

        # Wait for a connection
        print cl_cyan('Waiting for a connection...')
        self.connection, client_address = sock.accept()
        self.connection.settimeout(0.01)
        print cl_cyan('Connection from'), client_address
        self.isconnected = True
        last_read_time = time.time()

        while self.isconnected:
            try:
                msg = self.connection.recv(self.BUFFER_SIZE)
                last_read_time = time.time()    # Keep received time

                # Process the received msg pacage
                # parsing msg pack
                msg_splt = msg.split()

		if msg_splt[0]=='Joint_Pos' and len(msg_splt)==5:
		    self.JointPosition = [float(s) for s in msg_splt[1:]]


		if ( all(item != None for item in self.JointPosition) ):
		    self.isready = True
		else:
		    self.isready = False


            except:
                elapsed_time = time.time() - last_read_time
                if elapsed_time > 5.0:  # Didn't receive a pack in 5s
                    self.close() # Disconnect from iiwa
                    self.isconnected = False
                    print cl_lightred('No packet received from iiwa for 5s!')

        self.connection.shutdown(socket.SHUT_RDWR)
        self.connection.close()
        sock.close()
        self.isconnected = False
        print cl_lightred('Connection is closed!')
    #   ~M: Connection socket ===================

    #   M: Command send thread ==================
    # Each send command runs as a thread. May need to control the maximum running time (valid time to send a command).
    def send(self, cmd):
        thread.start_new_thread( self.__send, (cmd, ) )
    def __send(self, cmd):
        self.connection.sendall(cmd+'\r\n')
    #   ~M: Command send thread ==================

#   ~Class: Kuka iiwa TCP communication    #####################
#######################################################################################################################
class client:
    #   M: __init__ ===========================
    def __init__(self, ip, port):
        import socket

        self.BUFFER_SIZE = 1024
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.connect((ip, port))
        self.cmd = ''
        self.connected = True

        thread.start_new_thread( self.__send, () )

    def __send(self):
        while self.connected:
            self.soc.send(self.cmd)
            time.sleep(0.1)
    def close(self):
        self.connected =False
        self.soc.close()
######################################################################################################################

#   M:  Reading config file for Server IP and Port =================
def read_conf():
    f_conf = os.path.abspath(os.path.dirname(__file__)) + '/conf.txt'
    if os.path.isfile(f_conf):
        IP = ''
        Port = ''
        for line in open(f_conf, 'r'):
            l_splt = line.split()
            if len(l_splt)==4 and l_splt[0] == 'server':
                IP = l_splt[1]
                Port = int(l_splt[3])
        if len(IP.split('.'))!=4 or Port<=0:
            print cl_red('Error:'), "conf.txt doesn't include correct IP/Port! e.g. server 172.31.1.50 port 1234"
            exit()
    else:
        print cl_red('Error:'), "conf.txt doesn't exist!"
        exit()

    return [IP, Port]
#   ~M:  Reading config file for Server IP and Port ================


## MAIN ##
######################################################################################################################
if __name__ == '__main__':
    [IP, Port] = read_conf()

    miro_soc = server(IP, Port)

    #   Wait until iiwa is connected zzz!
    while (not miro_soc.isready): pass

    for i in range(1000):
        print miro_soc.JointPosition
        time.sleep(1)
######################################################################################################################
