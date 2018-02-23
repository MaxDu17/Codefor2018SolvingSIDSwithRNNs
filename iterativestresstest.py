from parse_data import DataParse as dp
class Source:
    class Native:
        INHALE_DIR = "dataSPLIT/inhale/"
        EXHALE_DIR = "datasplit/exhale/"
        UNKNOWN_DIR= "dataSPLIT/unknown/"
    class Server:
        INHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"
        EXHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"
        UNKNOWN_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"
dataprocess = dp()

for i in range(100):
    name = Source.Native.INHALE_DIR + str(i) + ".wav"
    try:
        dataprocess.load_wav_file(name)
    except:
        print("oops... check file: ", name)

for i in range(100):
    name = Source.Native.UNKNOWN_DIR + str(i) + ".wav"
    try:
        dataprocess.load_wav_file(name)
    except:
        print("oops... check file: ", name)

for i in range(100):
    name = Source.Native.EXHALE_DIR + str(i) + ".wav"
    try:
        dataprocess.load_wav_file(name)
    except:
        print("oops... check file: ", name)

print("all good!")