import serial
import time
arduino = serial.Serial(port='/dev/cu.usbmodem101', baudrate=115200, timeout=1) # TODO: change port maybe


def write_read(x_int): 
    print("write_read")
    arduino.write(bytes(str(x_int), 'utf-8')) 
    time.sleep(0.05) 
    data = arduino.readline()
    print(data) 

movements = {"up" : 2, "down" : 3, "stop": 4}



while True :
    user_input = input()
    if user_input not in ["up", "down"]:
        continue
    
    write_read(movements[user_input])