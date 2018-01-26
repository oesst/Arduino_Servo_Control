import time

import PyCmdMessenger

# ------------------------------------------------------------------------------
# Python program using the library to interface with the arduino sketch above.
# ------------------------------------------------------------------------------

# Initialize an ArduinoBoard instance.  This is where you specify baud rate and
# serial timeout.  If you are using a non ATmega328 board, you might also need
# to set the data sizes (bytes for integers, longs, floats, and doubles).
arduino = PyCmdMessenger.ArduinoBoard("/dev/ttyACM0", baud_rate=9600)

# List of command names (and formats for their associated arguments). These must
# be in the same order as in the sketch.
commands = [["connection", ""],
            ["is_connected", "i"],
            ["pan_angle", "i"],
            ["pan_angle_set", "i"],
            ["tilt_angle", "i"],
            ["tilt_angle_set", "i"],
            ["error", "s"]]

# Initialize the messenger
c = PyCmdMessenger.CmdMessenger(arduino, commands)

c.send('connection')
msg = c.receive()
print(msg)

c.send('tilt_angle', 0)
msg = c.receive()
print(msg)

time.sleep(3);

c.send('tilt_angle', 45)
msg = c.receive()
print(msg)

time.sleep(3);

c.send('tilt_angle', 90)
msg = c.receive()
print(msg)

time.sleep(3);

c.send('tilt_angle', 180)
msg = c.receive()
print(msg)

time.sleep(5);

c.send('tilt_angle', 0)
msg = c.receive()
print(msg)

