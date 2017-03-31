def record(WAVE_OUTPUT_FILENAME):
    import pyaudio
    import wave

    #WAVE_OUTPUT_FILENAME = 'output.wav'

    CHUNK = 1024*4
    FORMAT = pyaudio.paInt24
    CHANNELS = 8
    RATE = 48000
    RECORD_SECONDS = 11

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print '* Recording:', WAVE_OUTPUT_FILENAME

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print '* Done recording!'

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def play(WAVE_INPUT_FILENAME):
    import pyaudio
    import wave

    #WAVE_INPUT_FILENAME = 'chirp.wav'
    CHUNK = 1024

    wf = wave.open(WAVE_INPUT_FILENAME, 'rb')

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)

    while data != '':
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()

    p.terminate()


def chirp():
    import scipy.signal, scipy.io.wavfile
    import numpy as np
    import math, wave, pyaudio
    ###############################

    len = 10
    fre = 48000
    fMin = 10
    fMax = 20000

    t = np.arange(0, len, (1.0 / fre))

    y = scipy.signal.chirp(t, fMin, len, fMax, method='logarithmic', phi=-90, vertex_zero=True)
    ramp = [math.tanh(x)/2.0+0.5 for x in np.arange(-4,4,100.0/48000)]
    pad = np.zeros(48000)
    beep = scipy.signal.chirp(np.arange(0, 0.2, (1.0 / fre)), 600, 600, fMax, method='logarithmic', phi=-90, vertex_zero=True)
    beep[0:3840] = [beep[i] * ramp[i] for i in range(3840)]
    beep[-3840:] = [beep[-3840 + i] * (1 - ramp[i]) for i in range(3840)]

    y[0:3840] = [y[i] * ramp[i] for i in range(3840)]
    y[-3840:] = [y[-3840+i] * (1-ramp[i]) for i in range(3840)]
    sweep = np.concatenate((beep, pad, y, pad, beep), axis=0)

    scipy.io.wavfile.write('chirp.wav', fre, np.float32(sweep))


'''
from multiprocessing import Process
import time
#chirp()
#play('chirp.wav')
#record('output.wav')

for i in range(10):
    p1=Process(target = play, args=('test.wav',))
    p2=Process(target = record, args=('output'+str(i)+'.wav',))
    p1.start()
    p2.start()
    time.sleep(15)

#import random, math
#import matplotlib.pyplot as plt

#plt.plot([math.exp(i/48000.0) for i in range(48000)])
#plt.show()
'''