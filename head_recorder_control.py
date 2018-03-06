import time
import signal
from synced_recorder import SyncedRecorder
from servo_controller import ServoController

##############################################
### Initialize Connection to Arduino Board ###
##############################################

controller = ServoController();

##############################################
###       Initialize Sound Recorder        ###
##############################################

# Define recording time in seconds
RECORDING_TIME = 10
FILE_NAME_PREFIX = 'recordings/noise_low'
recorder = SyncedRecorder()

# Make sure we set the head back to the zero position before exiting
def reset_and_exit(signal, frame):
    print('Resetting Head and exiting program')
    controller.reset()
    exit(0)

# register the exiting function
signal.signal(signal.SIGINT, reset_and_exit)
signal.signal(signal.SIGSEGV, reset_and_exit)

##############################################
###         Define Head Locations          ###
##############################################
azimuth = [0, 10, 20, 30, -10, -20, -30]
elevations = [-30, -20, -10, 0, 10, 20, 30]

azimuth = [-90,-45,0,45,90]
azimuth = [0]
elevations = [0]

##############################################
###         Go in Recording Loop           ###
##############################################
controller.reset()




try:


    # controller.zeroing()
    for azi in azimuth:
        # bring head in correct position
        controller.set_azimuth(azi)

        for elev in elevations:
            # bring head in correct position
            controller.set_elevation(elev)
            # time.sleep(1) # <- no need to wait because recorder waits before recording automatically

            # recording at that position
            recorder.record(RECORDING_TIME)
            file_name = FILE_NAME_PREFIX + '_azi_' + str(azi) + '_ele_' + str(elev)
            recorder.save(file_name)
            recorder.finish()




    # shut down everything at the end
    recorder.finish()


except Exception as ex:
    print(ex)
    controller.reset()
    exit(0)

controller.reset()
