import wave
import struct
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    data_size = 8192
    fname = "100.wav"
    frate = 4096
    wav_file = wave.open(fname, 'r')
    data = wav_file.readframes(data_size)
    wav_file.close()
    data = struct.unpack('{n}h'.format(n=data_size), data)
    data = np.array(data)

    w = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(w))
    print(freqs.min(), freqs.max())
    # (-0.5, 0.499975)

    # Find the peak in the coefficients
    idx = np.argmax(np.abs(w))
    freq = freqs[idx]
    freq_in_hertz = abs(freq * frate)
    print(freq_in_hertz)
    w = np.abs(w[:20000])
    freq_span = np.linspace(0,11025/2,20000)
    #plt.plot(freqs*frate,np.abs(w) )
    plt.plot(freq_span[:5000],w[:5000])
    plt.show()
    # 439.8975