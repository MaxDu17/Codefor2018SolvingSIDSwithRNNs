import numpy as np
import wave
import struct



CHUNK = 44100
RECORDTIME = 10



def load_wav_file(name):
    # print(name)
    wav_file = wave.open(name, 'r')
    data = wav_file.readframes(RECORDTIME * CHUNK)
    print(len(data))
    wav_file.close()
    data = struct.unpack('{n}h'.format(n=RECORDTIME * CHUNK), data)
    data = np.array(data)
    return data


#data_blank = load_wav_file("noisereduction/blank.wav")
data_filled = load_wav_file("noisereduction/filledyet44100.wav")
#fourier_blank = np.fft.fft(data_blank)
fourier_filled = np.fft.fft(data_filled)

test = fourier_filled
#transformed = np.fft.ifft(test)
transformed = np.fft.ifft(np.fft.fft(data_filled))
transformed = transformed.astype('uint8')
print(len(transformed))
print(transformed)


wav_file = wave.open("noisereduction/filledyet44100.wav", 'r')
wf = wave.open("noisereduction/subtraction.wav", 'wb')

wf.setparams(wav_file.getparams())

for sample in transformed:
    value = struct.pack('h', sample)
    wf.writeframesraw(value)

wf.close()
