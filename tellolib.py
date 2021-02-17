import socket
import threading
import time
from stats import Stats
import re


class Tello:
  def __init__(self, tello_ip: str='192.168.10.1', debug: bool=True):
    # Opening local UDP port on 8889 for Tello communication
    self.local_ip = ''
    self.local_port = 8889
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.socket.bind((self.local_ip, self.local_port))

    self.local_port_state = 8890
    self.socket_state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.socket_state.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.socket_state.bind((self.local_ip, self.local_port_state))

    # Setting Tello ip and port info
    self.tello_ip = tello_ip
    self.tello_port = 8889
    self.tello_address = (self.tello_ip, self.tello_port)

    self.log = []
    self.state_log = []
    self.response = None

    # Intializing response thread
    self.receive_thread = threading.Thread(target=self._receive_thread)
    self.receive_thread.daemon = True
    self.receive_thread.start()

    # Intializing state thread
    self.state_thread = threading.Thread(target=self._state_thread)
    self.state_thread.daemon = True
    self.state_thread.start()

    # easyTello runtime options
    self.MAX_TIME_OUT = 2.0
    self.debug = debug

    # Setting Tello to command mode
    self.send_command('command')


  def __del__(self):
    self.socket.close()
    self.socket_state.close()


  def send_command(self, command: str, query: bool =False):
    # New log entry created for the outbound command
    self.log.append(Stats(command, len(self.log)))

    # Sending command to Tello
    self.socket.sendto(command.encode('utf-8'), self.tello_address)
    # Displaying conformation message (if 'debug' os True)
    if self.debug is True:
      print('Sending command (tellolib): {}'.format(command))

    # Checking whether the command has timed out or not (based on value in 'MAX_TIME_OUT')
    start = time.time()
    while not self.log[-1].got_response():  # Runs while no response has been received in log
      now = time.time()
      difference = now - start
      if difference > self.MAX_TIME_OUT:
        print('Connection timed out! (tellolib)')
        break
    # Prints out Tello response (if 'debug' is True)
    if self.debug is True and query is False:
      print('Response (tellolib): {}'.format(self.log[-1].get_response()))
  

  def _receive_thread(self):
    while True:
      # Checking for Tello response, throws socket error
      try:
        self.response, ip = self.socket.recvfrom(1024)
        self.log[-1].add_response(self.response)
      except socket.error as exc:
        print('Socket error (tellolib Receive): {}'.format(exc))


  def _state_thread(self):
    while True:
      # Checking for Tello State response, throws socket error
      try:
        self.state_response, ip = self.socket_state.recvfrom(1024)

        # Only add to log if it is different than previous
        self.state_response = self.format_state_response(self.state_response)

        if len(self.state_log) == 0 or self.state_response != self.state_log[-1]:
            self.state_log.append(self.state_response)
      except socket.error as exc:
        print('Socket error (tellolib State): {}'.format(exc))

      # Sample once a second
      time.sleep(1)

  
  # returns state data as a dict of key value pairs
  def format_state_response(self, state_response):
      state = re.split(r'[:;]\s*', state_response.decode("utf-8"))
      
      # Dict representation ("key": value) pairs
      # it = iter(state)
      # state = dict(zip(it, it))

      # Select just the "values"
      state = state[1::2]
      return state


  def wait(self, delay: float):
    # Displaying wait message (if 'debug' is True)
    if self.debug is True:
      print('Waiting {} seconds... (tellolib)'.format(delay))
    # Log entry for delay added
    self.log.append(Stats('wait', len(self.log)))
    # Delay is activated
    time.sleep(delay)


  def get_log(self):
    return self.log


  # Return current contents of state log. Clears log when called
  def get_state_log(self):
    tmp = self.state_log.copy()
    self.state_log.clear()
    return tmp

