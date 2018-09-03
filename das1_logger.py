import array
import time
import socket
import numpy as np
import soundfile as sf
import playsound as ps



### 2 Mic Recorder ###
######################


# This script simultaneously records sound from 2 input sources and stores it in two wav files
# Call it like that: python mic_sync_recorder.py name_of_recorded_files recording_time


class DAS1Logger:
    def __init__(self,playback_sound = []):

        self.HOST = 'localhost'
        self.PORT = 8997
        self.BUFFER_SIZE = 63000
        self.play_sound = False

        # if given play that sound
        if playback_sound:
            self.play_sound = True
            self.playback_sound = playback_sound
            self.logging_time = sf.info(playback_sound).duration
            self.RATE = sf.info(playback_sound).samplerate

    def connect_to_jAER(self):
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # reset timestamps
            # self.sock.sendto(b"zerotimestamps", (self.HOST, self.PORT))
            # print('Could not connect to socket : ', ce)


    def logging(self, logging_time,file_name):
        self.connect_to_jAER()

        if self.play_sound:
            logging_time = self.logging_time
            print("Play back sound %s and Logging %i seconds in ..." % (self.playback_sound,int(logging_time)))
        else:
            print("Logging %i seconds in ..." % int(logging_time))

        now = time.time()

        count = 2
        while count > 0:
            print(count)
            count -= 1
            time.sleep(1)



        if self.play_sound:
            ps.playsound(self.playback_sound,block=False)

        line = 'startlogging '+file_name
        line = bytes(line, 'utf-8')
        self.sock.sendto(line, (self.HOST, self.PORT))
        print('Logging Started...')


        # For debugging
        data, fromaddr = self.sock.recvfrom(self.BUFFER_SIZE)
        print('client received %r from %r' % (data, fromaddr))

        time.sleep(logging_time+0.1)

        print('Loggin Stopped')

        line = 'stoplogging'
        line = bytes(line, 'utf-8')
        self.sock.sendto(line, (self.HOST, self.PORT))
        self.close()

    def close(self):
        self.sock.close()


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print('Please provide exactly two arguments: recording time and file name e.g. python 2_mic_sync_recorder test 5')
        exit(1)

    recording_time = int(sys.argv[2])

    recorder = DAS1Logger()
    recorder.logging(recording_time,sys.argv[1])
    recorder.close()
    exit(0)
