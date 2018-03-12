import time
import signal

import os

from servo_controller import ServoController

##############################################
### Initialize Connection to Arduino Board ###
##############################################

controller = ServoController();

##############################################
###     Initialize Sound/DAS1 Recorder     ###
##############################################
RECORD_DAS1 = True

# Define recording time in seconds
RECORDING_TIME = 2
# If needed define path to playback sound
# ATTENTION: The duration of the recording is set to the duration of the playback_sound file !!
PLAYBACK_SOUND = '/home/oesst/cloudStore_UU/code_for_duc/noise_bursts.wav'
DIRECTORY = 'recordings/'
FILE_NAME_PREFIX = 'noise_low'

# choose what data to log. Do NOT forget to turn on UDP output in JAER
if RECORD_DAS1:
    from das1_logger import DAS1Logger
    recorder = DAS1Logger()
else:
    from synced_recorder import SyncedRecorder
    recorder = SyncedRecorder()



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

        azi_directory = 'azimuth_' + str(azi) + '/'
        # make sure directory exists
        os.makedirs(DIRECTORY+azi_directory, exist_ok=True)
        for elev in elevations:
            # bring head in correct position
            controller.set_elevation(elev)
            # time.sleep(1) # <- no need to wait because recorder waits before recording automatically

            time.sleep(1)


            save_path = DIRECTORY +azi_directory+ FILE_NAME_PREFIX + '_azi_' + str(azi) + '_ele_' + str(elev)

            if RECORD_DAS1:
                recorder.logging(RECORDING_TIME, save_path)
                recorder.close()
            else:
                # recording at that position
                recorder.record(RECORDING_TIME)
                recorder.save(save_path)
                recorder.finish()




    # shut down everything at the end
    recorder.finish()


except Exception as ex:
    print(ex)
    controller.reset()
    exit(0)

controller.reset()
