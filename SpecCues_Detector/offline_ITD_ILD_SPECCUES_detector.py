import struct
from scipy.io import loadmat
import wave

import numpy as np
import pyaudio
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from scipy.signal import welch, butter, lfilter, get_window
from scipy.ndimage.interpolation import rotate
import signal
import matplotlib.pyplot as plt
from som import SOM
from PyQt4 import QtTest
from servo_controller import ServoController

test_azi = 0
test_ele = -40

RIGHT_RECORDING = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/whiteNoise_1_20000Hz_normalEars/azimuth_' + str(test_azi) + '/whiteNoise_1_20000Hz_normalEars_azi_' + str(
    test_azi) + '_ele_' + str(test_ele) + '_right.wav'
LEFT_RECORDING  = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/whiteNoise_1_20000Hz_normalEars/azimuth_' + str(test_azi) + '/whiteNoise_1_20000Hz_normalEars_azi_' + str(
    test_azi) + '_ele_' + str(test_ele) + '_left.wav'

# RIGHT_RECORDING = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/sound_samples/vacuum_cleaner_normalEars/azimuth_' + str(test_azi) + '/vacuum_cleaner_normalEars_azi_' + str(
#     test_azi) + '_ele_' + str(test_ele) + '_right.wav'
# LEFT_RECORDING  = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/sound_samples/vacuum_cleaner_normalEars/azimuth_' + str(test_azi) + '/vacuum_cleaner_normalEars_azi_' + str(
#     test_azi) + '_ele_' + str(test_ele) + '_left.wav'


CUT_OFF_FREQ = 20000


# This script calculates ITD, ILD and Spectral Cues from real time data (2 microphones)
class RealTimeSpecAnalyzer(pg.GraphicsWindow):
    def __init__(self):
        super(RealTimeSpecAnalyzer, self).__init__(title="Live FFT")
        self.pa = pyaudio.PyAudio()

        ### Initialize Connection to Arduino Board

        # register the exiting function
        signal.signal(signal.SIGINT, self.reset_and_exit)
        signal.signal(signal.SIGSEGV, self.reset_and_exit)
        signal.signal(signal.SIGTERM, self.reset_and_exit)

        p = self.pa
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            print((i, dev['name'], dev['maxInputChannels']))

        # initialize SOM data
        self.som = SOM('som_data_L-R_L+R_whiteNoise_normaleEars_recordings.mat', 4)
        # self.som = SOM('som_data_L_averaged_recordings.mat', 4)

        # CONSTANTS
        self.RATE = 44100
        self.CHUNK_SIZE = 1024 * 8 # <- maybe change to 1024 (so that it is equal to the recording size, DOES NOT change PSD length)
        self.FORMAT = pyaudio.paInt16
        self.TIME = 2  # time period to display
        self.INTENS_THRES = -40  # intensity threshold for ILD & ITD in db
        self.counter = 0
        self.fft_bins = 128
        self.logScale = False  # display frequencies in log scale
        self.max_itd = 1.3  # determines the how precise the head follows the sound
        self.max_ild = 20  # determines the how precise the head follows the sound
        self.binCount = 30  # determines the resolution of the ITD values
        self.sum_angle = []
        self.skip_recording = False
        self.skip_recording_counter = 0
        self.show_spectrogram = False
        self.calc_spectral_cues = True
        self.azimuth_Detector_On = False

        # data storage
        self.data_l = np.zeros(self.RATE * self.TIME)
        self.data_l_tmp = np.zeros(self.RATE * self.TIME)
        self.data_r = np.zeros(self.RATE * self.TIME)
        self.data_r_tmp = np.zeros(self.RATE * self.TIME)
        self.frequencies_l = np.zeros(int(self.CHUNK_SIZE / 2))
        self.frequencies_r = np.zeros(int(self.CHUNK_SIZE / 2))
        self.itds = np.zeros(self.binCount)  # store only the recent 100 values
        self.ilds = np.zeros(self.binCount)  # store only the recent 100 values
        self.timeValues = np.linspace(0, self.TIME, self.TIME * self.RATE)
        self.img_array = -np.ones((500, int(self.fft_bins / 2)))
        self.response_SOM_grid = np.zeros((self.som.n_height, self.som.n_width))

        # set head azimuth angle
        self.azimuth = 0
        # noinspection PyCallByClass
        QtTest.QTest.qWait(2000)

        # initialization
        self.initMicrophones()
        self.initUI()

        # Timer to update plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        interval_ms = 1000 * (self.CHUNK_SIZE / self.RATE)
        print('Updating rate %.1f ms' % interval_ms)
        self.timer.start(interval_ms)

    # Make sure we set the head back to the zero position before exiting
    def reset_and_exit(self, signal, frame):
        print('Resetting Head and exiting program')
        self.controller.reset()
        exit(0)

    def initUI(self):
        # Setup plots
        self.setWindowTitle('ILD/ITD Analyzer')
        self.resize(1800, 800)

        # first plot, signals amplitude
        self.p1 = self.addPlot(colspan=2)
        self.p1.setLabel('bottom', 'Time', 's')
        self.p1.setLabel('left', 'Amplitude')
        self.p1.setTitle('')
        self.p1.setRange(xRange=(0, self.TIME), yRange=(-25000, 25000))
        # plot 2 signals in the plot
        self.ts_1 = self.p1.plot(pen=(1, 2))
        self.ts_2 = self.p1.plot(pen=(2, 2))

        if self.show_spectrogram:
            self.nextRow()
            # self.viewBox = self.addViewBox(colspan=2)
            # self.viewBox.setLabel('bottom', 'Time', 's')
            # self.viewBox.setLabel('left', 'Spectrum')
            ## lock the aspect ratio so pixels are always square
            # self.viewBox.setAspectLocked(False)

            ## Create image item
            self.pTest = self.addPlot(colspan=2)
            self.img = pg.ImageItem()
            self.pTest.addItem(self.img)
            self.pTest.setLabel('bottom', 'Time', 's')
            self.pTest.setLabel('left', 'Frequency Channel')
            # self.img.setLabel('left', 'Spectrum')
            # self.viewBox.addItem(self.img)
            # self.viewBox.setXRange(100,150)
            # self.viewBox.setYRange(500, 100)
            # self.viewBox.addItem(self.labelItem)

            color = plt.cm.afmhot
            colors = color(range(0, 256))[:]
            pos = np.linspace(0, 1, 256)
            cmap = pg.ColorMap(pos, colors)
            lut = cmap.getLookupTable(0.0, 1.0, 256)

            self.img.setLookupTable(lut)
            self.img.setLevels([-1, 0])
            # self.img.scale(10,   10.5)
            # self.win = np.hanning(self.CHUNK_SIZE)

        if self.calc_spectral_cues:
            self.nextRow()
            # self.viewBox = self.addViewBox(colspan=2)
            # self.viewBox.setLabel('bottom', 'Time', 's')
            # self.viewBox.setLabel('left', 'Spectrum')
            ## lock the aspect ratio so pixels are always square
            # self.viewBox.setAspectLocked(False)

            ## Create image item
            self.pSOM = self.addPlot(colspan=2)
            self.imgSOM = pg.ImageItem()
            self.pSOM.addItem(self.imgSOM)
            self.pSOM.setLabel('bottom', 'Azimuth', 'deg')
            self.pSOM.setLabel('left', 'Elevation', 'deg')
            # self.viewBox.addItem(self.img)
            # self.viewBox.setXRange(100,150)
            # self.viewBox.setYRange(500, 100)
            # self.viewBox.addItem(self.labelItem)

            color = plt.cm.afmhot
            colors = color(range(0, 256))[:]
            pos = np.linspace(0, 1, 256)
            cmap = pg.ColorMap(pos, colors)
            lut = cmap.getLookupTable(0.0, 1.0, 256)

            self.imgSOM.setLookupTable(lut)
            self.imgSOM.setLevels([0, 1])
            # self.img.scale(10,   10.5)
            # self.win = np.hanning(self.CHUNK_SIZE)

        # # frequency of signal 1
        # self.p2 = self.addPlot(colspan=2)
        # self.p2.setLabel('bottom', 'Frequency LEFT', 'Hz')
        # self.spec_left = self.p2.plot(pen=(50, 100, 200),
        #                               brush=(50, 100, 200),
        #                               fillLevel=-100)
        # if self.logScale:
        #     self.p2.setRange(xRange=(0, 15000),
        #                      yRange=(-60, 20))
        #     self.spec_left.setData(fillLevel=-100)
        #     self.p2.setLabel('left', 'PSD', 'dB / Hz')
        # else:
        #     self.p2.setRange(xRange=(0, 15000),
        #                      yRange=(0, 50))
        #     self.spec_left.setData(fillLevel=0)
        #     self.p2.setLabel('left', 'PSD', '1 / Hz')
        #
        # self.nextRow()
        #
        # # frequency of signal 2
        # self.p3 = self.addPlot(colspan=2)
        # self.p3.setLabel('bottom', 'Frequency RIGHT', 'Hz')
        # self.spec_right = self.p3.plot(pen=(50, 100, 200),
        #                                brush=(50, 100, 200),
        #                                fillLevel=-100)
        # if self.logScale:
        #     self.p3.setRange(xRange=(0, 15000),
        #                      yRange=(-60, 20))
        #     self.spec_right.setData(fillLevel=-100)
        #     self.p3.setLabel('left', 'PSD', 'dB / Hz')
        # else:
        #     self.p3.setRange(xRange=(0, 15000),
        #                      yRange=(0, 50))
        #     self.spec_right.setData(fillLevel=0)
        #     self.p3.setLabel('left', 'PSD', '1 / Hz')

        # write ITD & ILD in a box
        # self.viewBox = self.addViewBox(colspan=2)
        # self.textITD = pg.TextItem(text='The ITD is 0.0 ms', anchor=(-1.0, 6.0), border='w', fill=(255, 255, 255), color='#000000')
        # self.viewBox.addItem(self.textITD)
        # self.textILD = pg.TextItem(text='The ILD is 0.0 ', anchor=(-1.0, 5.0), border='w', fill=(255, 255, 255), color='#000000')
        # self.viewBox.addItem(self.textILD)

        self.nextRow()

        # plot ITD and ILD bins
        p4 = self.addPlot(row=4, col=0, rowspan=1, colspan=1)
        p4.setRange(xRange=(-1.5, 1.5))
        self.histo_itd = p4.plot(stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        p4.setLabel('bottom', 'ITD', '')
        p4.setLabel('left', 'Belief')

        p5 = self.addPlot(row=4, col=1, rowspan=1, colspan=1)
        p5.setRange(xRange=(-10, 10))
        self.histo_ild = p5.plot(stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        p5.setLabel('bottom', 'ILD', '')
        p5.setLabel('left', 'Belief')

    def initMicrophones(self):
        print('Reading data from file ...')
        self.stream_l = wave.open(LEFT_RECORDING, 'rb')
        self.stream_r = wave.open(RIGHT_RECORDING, 'rb')

    def readData(self):
        # read data of first device
        try:
            block = self.stream_l.readframes(self.CHUNK_SIZE)
            count = len(block) / 2
            format = '%dh' % (count)
            data_l = struct.unpack(format, block)

            block = self.stream_r.readframes(self.CHUNK_SIZE)
            count = len(block) / 2
            format = '%dh' % (count)
            data_r = struct.unpack(format, block)

            if len(data_l) <= 0 or len(data_r) <= 0:
                self.stream_r.rewind()
                self.stream_l.rewind()

                block = self.stream_l.readframes(self.CHUNK_SIZE)
                count = len(block) / 2
                format = '%dh' % (count)
                data_l = struct.unpack(format, block)

                block = self.stream_r.readframes(self.CHUNK_SIZE)
                count = len(block) / 2
                format = '%dh' % (count)
                data_r = struct.unpack(format, block)


        except (IOError,ValueError) as ioerr:
            self.stream_r.rewind()
            self.stream_l.rewind()




        return np.array(data_l), np.array(data_r)

    def overlap(self, x, window_size, window_step):
        """
        Create an overlapped version of X
        Parameters
        ----------
        X : ndarray, shape=(n_samples,)
            Input signal to window and overlap
        window_size : int
            Size of windows to take
        window_step : int
            Step size between windows
        Returns
        -------
        X_strided : shape=(n_windows, window_size)
            2D array of overlapped X
        """
        if window_size % 2 != 0:
            raise ValueError("Window size must be even!")
        # Make sure there are an even number of windows before stridetricks
        append = np.zeros((window_size - len(x) % window_size))
        x = np.hstack((x, append))

        ws = window_size
        ss = window_step
        a = x

        valid = len(a) - ws
        nw = (valid) // ss
        out = np.ndarray((int(nw), int(ws)), dtype=a.dtype)

        for i in np.arange(nw):
            # "slide" the window along the samples
            start = int(i * ss)
            stop = int(start + ws)
            out[int(i)] = a[start:stop]

        return out

    def stft(self,
             x,
             fftsize=128,
             step=65,
             mean_normalize=True,
             real=False,
             compute_onesided=True):
        """
        Compute STFT for 1D real valued input X
        """
        if real:
            local_fft = np.fft.rfft
            cut = -1
        else:
            local_fft = np.fft.fft
            cut = None
        if compute_onesided:
            cut = fftsize // 2
        if mean_normalize:
            x -= x.mean()

        x = self.overlap(x, fftsize, step)

        size = fftsize
        win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
        x = x * win[None]
        x = local_fft(x)[:, :cut]
        return x

    def pretty_spectrogram(self, d, log=True, thresh=5, fft_size=512, step_size=64):

        """
        creates a spectrogram
        log: take the log of the spectrgram
        thresh: threshold minimum power for log spectrogram
        """
        specgram = np.abs(
            self.stft(
                d,
                fftsize=fft_size,
                step=step_size,
                real=False,
                compute_onesided=True))

        if log == True:
            specgram /= specgram.max()  # volume normalize to max 1
            specgram = np.log10(specgram)  # take log
            specgram[specgram < -thresh] = -thresh  # set anything less than the threshold as the threshold
        else:
            specgram[specgram < thresh] = thresh  # set anything less than the threshold as the threshold

        return specgram

    def get_spectrum(self, data):
        T = 1.0 / self.RATE
        N = data.shape[0]
        Pxx = (1. / N) * np.fft.rfft(data)
        f = np.fft.rfftfreq(N, T)
        return np.array(f[1:].tolist()), np.array((np.absolute(Pxx[1:])).tolist())

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=6):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def cross_correlation_using_fft(self, x, y):
        from numpy.fft import fft, ifft, fft2, ifft2, fftshift

        f1 = fft(x)
        f2 = fft(np.flipud(y))
        cc = np.real(ifft(f1 * f2))
        return fftshift(cc)

    def get_welch_spectrum(self, data):
        # hanning = get_window('hamming',175)
        # f, psd = welch(data, RATE, hanning,detrend=False)
        fs = 128  # change that so that it is equal to the readin matlab file
        window = get_window('hann', fs * 1)
        f, psd = welch(data, fs=self.RATE, window=window, nfft=fs * 2, noverlap=int(fs / 2), axis=0, scaling="density", detrend=False)

        return f, psd

    def find_delay(self, a, b, max_delay=0):
        # very accurate but not so fast as gcc
        # from scipy.signal import correlate
        # corr = correlate(a, b)
        # corr = np.correlate(a,b,'full')
        corr = self.cross_correlation_using_fft(a, b)
        # check only lags that are in range -max_delay and max_delay
        # print(corr)
        if max_delay:
            middle = np.int(np.ceil(len(corr) / 2))
            new_corr = np.zeros(len(corr))
            new_corr[middle - max_delay:middle + max_delay] = corr[middle - max_delay:middle + max_delay]
            lag = np.argmax(np.abs(new_corr)) - np.floor(len(new_corr) / 2)
        else:
            lag = np.argmax(np.abs(corr)) - np.floor(len(corr) / 2)

        return lag

    def update(self):

        data_l, data_r = self.readData()
        # simulate input


        self.data_r_tmp = np.roll(self.data_r_tmp, -self.CHUNK_SIZE)
        self.data_r_tmp[-self.CHUNK_SIZE:] = data_r

        self.data_l_tmp = np.roll(self.data_l_tmp, -self.CHUNK_SIZE)
        self.data_l_tmp[-self.CHUNK_SIZE:] = data_l

        # only record when servos are not moving
        if not self.skip_recording:

            self.data_r = np.roll(self.data_r, -self.CHUNK_SIZE)
            self.data_r[-self.CHUNK_SIZE:] = data_r

            self.data_l = np.roll(self.data_l, -self.CHUNK_SIZE)
            self.data_l[-self.CHUNK_SIZE:] = data_l

        else:



            # plot data in any case. with that we prevent the stop of the plotting
            self.ts_1.setData(x=self.timeValues, y=self.data_l_tmp)
            self.ts_2.setData(x=self.timeValues, y=self.data_r_tmp)
            # self.spec_left.setData(x=f_l, y=(20 * np.log10(psd_l) if self.logScale else psd_l))
            # self.spec_right.setData(x=f_l, y=(20 * np.log10(psd_r) if self.logScale else psd_r))

            if self.show_spectrogram:
                psd = self.pretty_spectrogram(
                    data_l.astype('float64'),
                    log=True,
                    thresh=1,
                    fft_size=self.fft_bins,
                    step_size=self.CHUNK_SIZE)

                # roll down one and replace leading edge with new data

                self.img_array = np.roll(self.img_array, -1, 0)
                self.img_array[-1:] = psd
                self.img.setImage(self.img_array, autoLevels=False)

            # set back skip recording var
            if self.skip_recording_counter > 2:
                self.skip_recording = False
                self.skip_recording_counter = 0
            else:
                self.skip_recording_counter += 1
                return




        # get frequency spectrum
        f_l, psd_l = self.get_welch_spectrum(data_l)
        f_r, psd_r = self.get_welch_spectrum(data_r)

        # make log scale
        psd_l = np.log10(psd_l) * 20
        psd_r = np.log10(psd_r) * 20

        # cut off frequencies at the end (>18000Hz)
        indis = f_l < CUT_OFF_FREQ

        f_l = f_l[indis]
        f_r = f_r[indis]
        psd_l = psd_l[indis]
        psd_r = psd_r[indis]

        # store frequency spectrum data for later use
        # self.frequencies_l += psd_l
        # self.frequencies_r += psd_r

        # print(psd_l.shape) # -> is 44
        # print(psd_r.shape) # -> is 44

        # filter the signal
        # data_l = self.butter_bandpass_filter(data_l, 100, 15000, self.RATE, order=2)
        # data_r = self.butter_bandpass_filter(data_r, 100, 15000, self.RATE, order=2)

        # get amplitude values in dB (theoretically it is not dB, since we don't have anything to compare to)
        signal_l = data_l / 2 ** 15
        signal_r = data_r / 2 ** 15
        intensities_l = np.log10(np.abs(signal_l)) * 20.0
        intensities_r = np.log10(np.abs(signal_r)) * 20.0

        # if any intensity exceeds as threshold calculate ITD and ILD
        if any(intensities_l > self.INTENS_THRES) or any(intensities_r > self.INTENS_THRES):

            # if counter is bigger than 100 -> reset it to 0
            if (self.counter >= self.binCount):
                self.counter = 0

            # # calculate ILD, use only frequencies between 1500 and 10000 Hz (indicies 138 til 927)
            ILD = 10 * np.log10(np.sum(np.abs(signal_l) ** 2) / np.sum(np.abs(signal_r) ** 2))
            # store values in counter index -> only recent 100 values
            self.ilds[self.counter] = ILD - 1.5

            # calculate ITD, use only frequencies between 100 and 1000 Hz (indicies 8 til 91)
            signal_itd_l = data_l
            signal_itd_r = data_r
            # np.lib.pad(signal_itd_l, (100, 0), 'constant', constant_values=(0, 0)), 'same')
            ITD = self.find_delay(signal_itd_l, signal_itd_r, 100) / self.RATE
            # store values in counter index -> only recent 100 values
            self.itds[self.counter] = (ITD * 1000)

            # calcualte location based on spectral cues
            if self.calc_spectral_cues:
                # TODO be careful to use the correct data as an input e.g. [PSDL,PSDR] or [PSDR/PSDR,ITD,ILD]


                # rescale
                tmp1 = psd_l - psd_r
                tmp1 += - (np.min(tmp1))
                tmp1 /= np.max(tmp1)

                tmp2 = psd_l + psd_r
                tmp2 += - (np.min(tmp2))
                tmp2 /= np.max(tmp2)

                tmp = np.concatenate((tmp1, tmp2))

                # tmp1 = psd_l
                # tmp1 += - (np.min(tmp1))
                # tmp = tmp1 / np.max(tmp1)


                f_perceived, winning_node_activity, winning_node_inds = self.som.calculate_location(tmp)
                trace = 0.99

                # normalize
                activity = self.som.response_activity
                # activity += -(np.min(activity[np.nonzero(activity)]))
                # activity /= np.max(activity)

                # create a grid
                activity[:] = 0.0
                activity[winning_node_inds] = 1.0
                # activity = self.som.response_activity
                # activity[activity<0.75] = 0.0
                activity = np.reshape(activity, (self.som.n_height, self.som.n_width))
                # make a smooth transition between the received data points
                self.response_SOM_grid = trace * self.response_SOM_grid + (1 - trace) * activity
                # set the new image
                self.imgSOM.setImage(self.response_SOM_grid.T, autoLevels=False, levels=(0.0,1))

                #print(activity)

            # update textbox
            # self.textILD.setPlainText('The ILD is %f ' % ILD)
            # self.textITD.setPlainText('The ITD is %f ms' % (ITD * 1000))
            if self.counter % 10 == 0:
                print('The ITD is %f ms and the ILD is %f. Control angle : %f' % ((ITD * 1000), ILD, self.azimuth))

            # plot ITD as histogram
            y, x = np.histogram(self.itds, bins=np.linspace(-self.max_itd, self.max_itd, 180))
            self.histo_itd.setData(x=x, y=y)
            # plot ILD as histogram
            y, x = np.histogram(self.ilds, bins=np.linspace(-25, 25, 500))
            self.histo_ild.setData(x=x, y=y)

            # increase counter
            self.counter += 1

        # plot data
        self.ts_1.setData(x=self.timeValues, y=self.data_l_tmp)
        self.ts_2.setData(x=self.timeValues, y=self.data_r_tmp)
        # self.spec_left.setData(x=f_l, y=(20 * np.log10(psd_l) if self.logScale else psd_l))
        # self.spec_right.setData(x=f_l, y=(20 * np.log10(psd_r) if self.logScale else psd_r))

        if self.show_spectrogram:
            psd = self.pretty_spectrogram(
                data_l.astype('float64'),
                log=True,
                thresh=1,
                fft_size=self.fft_bins,
                step_size=self.CHUNK_SIZE)

            # roll down one and replace leading edge with new data

            self.img_array = np.roll(self.img_array, -1, 0)
            self.img_array[-1:] = psd
            self.img.setImage(self.img_array, autoLevels=False)

        if self.azimuth_Detector_On:

            ###########################################
            ###          HEAD CONTROLLER            ###
            ###########################################

            # # determine max ITD
            # ITD = np.histogram(self.itds, bins=np.linspace(-self.max_itd, self.max_itd, 180))
            # ILD = np.histogram(self.ilds, bins=np.linspace(-self.max_ild, self.max_ild, 180))

            # y,x = np.histogram(self.ilds, bins=np.linspace(-self.max_ild, self.max_ild, 500))
            # ILD = np.argmax(y)-250

            histCount = 180

            ########################
            ###        ILD       ###
            ########################
            # maybe the offset depends actually on the noise? or background noise ?
            offset = 2
            y, x = np.histogram(self.ilds, bins=np.linspace(-self.max_ild, self.max_ild, histCount))
            angle_ILD = -int(np.argmax(y) - histCount / 2) + offset

            sd_ILD = np.std(y)

            # # maybe for the ILD use a fix value instead of the angle?
            # if np.abs(angle_ILD) > 2:
            #     self.azimuth = self.azimuth + angle_ILD*0.05

            ########################
            ###        ITD       ###
            ########################

            y, x = np.histogram(self.itds, bins=np.linspace(-self.max_itd, self.max_itd, histCount))
            angle_ITD = int(np.argmax(y) - histCount / 2)
            sd_ITD = np.std(y)

            # # turn the head to ensure that the itd has smallest possible value
            # if np.abs(angle_ITD) > 2:
            #     self.azimuth = self.azimuth + angle_ITD*0.05

            # calculate final angle to look at. TODO use a weighting function for ITD and ILD
            # first try...
            angle = (angle_ILD * sd_ILD + angle_ITD * sd_ITD) / (sd_ILD + sd_ITD)

            # print(sd_ITD,'',sd_ILD)

            # turn the head to ensure that the itd has smallest possible value
            if np.abs(angle) > 2:
                self.azimuth = self.azimuth + angle * 0.05

            # just to be sure that we stay in the range...
            if self.azimuth > 90:
                self.azimuth = 90
            elif self.azimuth < -90:
                self.azimuth = -90

            if self.counter % 5 == 0:
                self.controller.set_azimuth(self.azimuth)

                # the servos are moving now, so skip the recordings
                self.skip_recording = True

    def closeEvent(self, event):
        self.controller.reset()
        self.stream.close()
        self.pa.terminate()
        event.accept()  # let the window close


# QtGui.QApplication.setGraphicsSystem('opengl')
app = QtGui.QApplication([])

win = RealTimeSpecAnalyzer()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
