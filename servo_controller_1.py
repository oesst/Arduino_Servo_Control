import time

import PyCmdMessenger


# Initialize the board and baud_rate
arduino = PyCmdMessenger.ArduinoBoard("/dev/ttyACM0", baud_rate=9600)


commands = [["connection", ""],
            ["is_connected", "i"],
            ["pan_angle", "i"],
            ["pan_angle_set", "i"],
            ["tilt_angle", "i"],
            ["tilt_angle_set", "i"],
            ["error", "s"]]

# Initialize messenger
c = PyCmdMessenger.CmdMessenger(arduino, commands)

c.send('connection')

#
# msg = c.receive()
# print(msg)
# c.send('pan_angle', 0)
# msg = c.receive()
# print(msg)
# time.sleep(4)
# c.send('pan_angle', 90)
# msg = c.receive()
# print(msg)
# time.sleep(6)
# c.send('pan_angle', 180)
# msg = c.receive()
# print(msg)

c.send('tilt_angle', 0)
msg = c.receive()
print(msg)
time.sleep(5);

for i in range(45,270,45):
    c.send('tilt_angle', i)
    msg = c.receive()
    print(msg)
    time.sleep(3);

c.send('tilt_angle', 0)
msg = c.receive()
print(msg)
time.sleep(5)
