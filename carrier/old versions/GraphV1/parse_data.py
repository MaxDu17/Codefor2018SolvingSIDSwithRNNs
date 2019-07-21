import numpy as np
import matplotlib.pyplot as plt
import struct
import wave

CHUNK = 4096
RECORDTIME = 2



class DataParse:

    def load_wav_file(self,name):
        #print(name)
        wav_file = wave.open(name, 'r')
        data = wav_file.readframes(RECORDTIME*CHUNK)
        wav_file.close()
        data = struct.unpack('{n}h'.format(n=RECORDTIME*CHUNK), data)
        data = np.array(data)
        return data


    def split_file(self,data):

        time_split = list()
        for i in range(16):
            current_block_start = i*512
            current_block_end = current_block_start + 512
            time_split.append(data[current_block_start:current_block_end])
        return time_split


    def load_fourier(self,data):

        fourier_output = np.abs(np.fft.fft(data))
        fourier_output = fourier_output[1:int(CHUNK/16)]#truncates according to nyquist limit
        if np.max(fourier_output) == 0:
            fourier_output_norm = fourier_output
        else:
            fourier_output_norm = (fourier_output - np.min(fourier_output)) / (np.max(fourier_output) - np.min(fourier_output))
        frequency_division = np.linspace(1,CHUNK/2, int((CHUNK/16)-1))
        return fourier_output_norm, frequency_division

    def bin(self,output):
        output_bins = [None]*43

        for i in range(42):
            first = i*6
            last = first+6
            output_bins[i] = np.mean(output[first:last])
        output_bins[42] = np.mean(output[42*6:])
        return output_bins

    def prepare_data(self,name):
        data_list = list()
        data = self.load_wav_file(name)
        time_split = self.split_file(data)
        for time_slice in time_split:
            output, frq_div = self.load_fourier(time_slice)
            output_bins = self.bin(output)
            data_list.append(output_bins)
        return data_list

    def bins_from_stream(self,data):
        data_list = list()
        time_split = self.split_file(data)
        time_split = time_split[0:16]
        for time_slice in time_split:
            output, frq_div = self.load_fourier(time_slice)
            output_bins = self.bin(output)
            data_list.append(output_bins)
        return data_list


