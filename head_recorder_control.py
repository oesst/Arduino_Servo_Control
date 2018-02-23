import time
from mic_sync_recorder import SyncedRecorder
from servo_controller import ServoController
import PyCmdMessenger

##############################################
### Initialize Connection to Arduino Board ###
##############################################

controller = ServoController();

##############################################
###       Initialize Sound Recorder        ###
##############################################
recorder = SyncedRecorder();

# Define recording time in seconds
RECORDING_TIME = 2
FILE_NAME_PREFIX = 'recordings/white_noise_60db'
recorder = SyncedRecorder()

##############################################
###         Define Head Locations          ###
##############################################
azimuth = [0, 10, 20, 30, -10, -20, -30]
elevations = [-30, -20, -10, 0, 10, 20, 30]

azimuth = [90]
elevations = [-90,0,180]

##############################################
###         Go in Recording Loop           ###
##############################################
controller.reset()
git


controller.set_azimuth(0)
input('Confirm!')
controller.set_azimuth(90)

# time.sleep(3)
# controller.set_azimuth(0)
# time.sleep(3)
# controller.set_azimuth(90)
controller.reset()


# controller.zeroing()
# for azi in azimuth:
#     # bring head in correct position
#     controller.set_azimuth(azi)
#     time.sleep(1.5)
#
#     for elev in elevations:
#         # bring head in correct position
#         controller.set_elevation(elev)
#         # time.sleep(1) # <- no need to wait because recorder waits before recording automatically
#
#         # recording at that position
#         recorder.record(RECORDING_TIME)
#         file_name = FILE_NAME_PREFIX + '_azi_' + str(azi) + '_ele_' + str(elev)
#         recorder.save(file_name)
#         recorder.finish()
#
# # shut down everything at the end
# recorder.finish()
# controller.reset()


# c.send('tilt_angle', 0)
# msg = c.receive()
# print(msg)
# time.sleep(5);
#
# for i in range(45,270,45):
#     c.send('tilt_angle', i)
#     msg = c.receive()
#     print(msg)
#     time.sleep(3);
#
# c.send('tilt_angle', 0)
# msg = c.receive()
# print(msg)
# time.sleep(5)
