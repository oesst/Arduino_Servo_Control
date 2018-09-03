import time

import PyCmdMessenger


class ServoController:
    commands = [["connection", ""],
                ["is_connected", "i"],
                ["pan_angle", "i"],
                ["pan_angle_set", "i"],
                ["pan_angle_fast", "i"],
                ["tilt_angle", "i"],
                ["tilt_angle_set", "i"],
                ["error", "s"]]

    def __init__(self):
        # Initialize the board and baud_rate
        arduino = PyCmdMessenger.ArduinoBoard("/dev/ttyACM0", baud_rate=9600)
        # arduino = PyCmdMessenger.ArduinoBoard("/dev/ttyUSB0", baud_rate=9600)
        # Initialize messenger
        self.controller = PyCmdMessenger.CmdMessenger(arduino, ServoController.commands)
        self.controller.send('connection')
        msg = self.controller.receive()
        print(msg)

    # sets all servos to middle angle so that we can adjust the head model
    def zeroing(self):
        # set head to zero position
        self.set_elevation(0)
        self.set_azimuth(0)

        # wait until zeroing of head is confirmed
        # input("Press Enter to confirm zeroing...")

    # sets all servos to zero control angle
    def reset(self):
        # set head to zero position
        print('Resetting servos to 0 degree control angle...')
        self.set_elevation(180)
        self.set_azimuth(-90)
        print('Resetting done!')


    def set_elevation(self, elevation):
        # We use a 270 degree servo control angle for elevation.
        # 0 degree elevation corresponds to 90 degree control angel of the servo, since the head can have higher positive elevation than negative ones
        # elevation angle needs to be in range [-90,180]
        if -90 <= elevation <= 180:
            elevation =elevation +90
            angle = 270 - elevation

            self.controller.send('tilt_angle', int(angle))
            self.receive_msg()
        else:
            print('Given angle not in controllable range... Abort!')

    def set_azimuth(self, azimuth):
        # We use a 180 degree servo control angle for azimuth. That means that 0 degree azimuth corresponds to 90 degree control angle of the servo

        # azimuth angle needs to be in range [-90,90]
        if -90 <= azimuth <= 90:
            angle = azimuth + 90
            self.controller.send('pan_angle', int(angle))
            # self.receive_msg()
        else:
            print('Given angle not in controllable range... Abort!')

    def set_azimuth_fast(self, azimuth):
        # We use a 180 degree servo control angle for azimuth. That means that 0 degree azimuth corresponds to 90 degree control angle of the servo

        # azimuth angle needs to be in range [-90,90]
        if -90 <= azimuth <= 90:
            angle = azimuth + 90
            self.controller.send('pan_angle_fast', angle)
            # self.receive_msg()
        else:
            print('Given angle not in controllable range... Abort!')

    # goes into a loop and waits for a message that confirms the commnd
    def receive_msg(self):
        msg = []

        while not msg:
            msg = self.controller.receive()
            time.sleep(0.1)

        print(msg)
