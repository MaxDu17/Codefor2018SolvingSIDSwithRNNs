import numpy as np
import matplotlib.pyplot as plt
import struct
import wave

CHUNK = 4096 #to be changed
RECORDTIME = 2
THRESHOLDVALUE = 500


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

    def normalize(self,data):
        if np.max(data) == 0:
            data_out = data + 0.01
        else:
            data_out = (data-np.min(data))/(np.max(data)-np.min(data))
        return data_out

    def truncate_lower(self,data):
        new_data = list()
        for sample in data:
            new_data.append(sample[3:43])
        return new_data

    def load_fourier(self,data):

        fourier_output = np.abs(np.fft.fft(data))
        fourier_output = fourier_output[1:int(CHUNK/16)]#truncates according to nyquist limit
        frequency_division = np.linspace(1,CHUNK/2, int((CHUNK/16)-1))
        return fourier_output, frequency_division

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
        data_list = self.normalize(data_list)
       # data_list = self.truncate_lower(data_list)
        return data_list

    def prepare_data_autoencoder(self,name):
        data_list = list()
        data = self.load_wav_file(name)
        time_split = self.split_file(data)
        for time_slice in time_split:
            output, frq_div = self.load_fourier(time_slice)
            data_list.append(output)
        data_list = self.normalize(data_list)
        return data_list

    def bins_from_stream(self,data):
        data_list = list()
        time_split = self.split_file(data)
        time_split = time_split[0:16]
        for time_slice in time_split:
            output, frq_div = self.load_fourier(time_slice)
            output_bins = self.bin(output)
            data_list.append(output_bins)
        data_list = self.normalize(data_list)
       # data_list = self.truncate_lower(data_list)
        return data_list


