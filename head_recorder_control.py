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
RECORD_DAS1 = False

# Define recording time in seconds
RECORDING_TIME = 2
# If needed define path to playback sound
# ATTENTION: The duration of the recording is set to the duration of the playback_sound file !!
cwd = os.getcwd()
PLAYBACK_SOUND = os.path.join(cwd,'playback_sounds','white_noise_1_20000_hz_2000ms.wav')
DIRECTORY = os.path.join(cwd,'recordings_free_field')
FILE_NAME_PREFIX = 'free_field'

# choose what data to log. Do NOT forget to turn on UDP output in JAER
if RECORD_DAS1:
    from das1_logger import DAS1Logger
    if PLAYBACK_SOUND:
        recorder = DAS1Logger(PLAYBACK_SOUND)
    else:
        recorder = DAS1Logger()
else:
    from synced_recorder import SyncedRecorder
    if PLAYBACK_SOUND:
        recorder = SyncedRecorder(PLAYBACK_SOUND)
    else:
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

azimuth = [-90,0,90]
elevations = [0]

##############################################
###         Go in Recording Loop           ###
##############################################
controller.reset()

print('Recording process starts in 20s....')
time.sleep(1)

# try:

    # controller.set_elevation(0)
    #
    # controller.set_azimuth(-90)
    # time.sleep(5)
    # controller.set_azimuth(-45)
    # time.sleep(5)
    # controller.set_azimuth(0)
    # time.sleep(5)
    # controller.set_azimuth(45)
    # time.sleep(5)
    # controller.set_azimuth(90)
    # time.sleep(5)

# controller.zeroing()
for azi in azimuth:
    # bring head in correct position
    controller.set_azimuth_fast(azi)

    azi_directory = os.path.join(DIRECTORY,'azimuth_' + str(azi))
    # make sure directory exists
    os.makedirs(azi_directory, exist_ok=True)
    for elev in elevations:
        # bring head in correct position
        controller.set_elevation(elev)
        # time.sleep(1) # <- no need to wait because recorder waits before recording automatically

        save_path = os.path.join(azi_directory, FILE_NAME_PREFIX + '_azi_' + str(azi) + '_ele_' + str(elev))

        if RECORD_DAS1:
            recorder.logging(RECORDING_TIME, save_path)
            recorder.close()
        else:
            # recording at that position
            recorder.record(RECORDING_TIME)
            recorder.save(save_path)
            recorder.finish()



if not RECORD_DAS1:
    # shut down everything at the end
    recorder.finish()


# except (KeyboardInterrupt,Exception) as ex:
#     print(ex)
#     controller.reset()
#     exit(0)

controller.reset()
