import time
import signal

import os

from servo_controller import ServoController
import numpy as np
##############################################
### Initialize Connection to Arduino Board ###
##############################################

controller = ServoController();



# Make sure we set the head back to the zero position before exiting
def reset_and_exit(signal, frame):
    print('Resetting Head and exiting program')
    controller.reset()
    exit(0)

# register the exiting function
signal.signal(signal.SIGINT, reset_and_exit)
signal.signal(signal.SIGSEGV, reset_and_exit)
signal.signal(signal.SIGTERM, reset_and_exit)


##############################################
###         Go in Recording Loop           ###
##############################################
# controller.reset()
# controller.reset()
#
#
time.sleep(0)
#
# controller.set_azimuth(90)
#
# time.sleep(8)
#
# controller.reset()

#
# controller.calibrate()
#
time.sleep(2)

for i in np.arange(-90,90,1Caepten98
5):
    controller.set_elevation(int(i))
    time.sleep(1)



time.sleep(1)

controller.reset()
