from parse_data import DataParse as dp

class Source:
    class Native:
        INHALE_DIR = "sen_data/inhale/"
        EXHALE_DIR = "sen_data/exhale/"
        UNKNOWN_DIR= "sen_data/unknown/"
    class Server:
        INHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/sen_data/inhale/"
        EXHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/sen_data/inhale/"
        UNKNOWN_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/sen_data/inhale/"
dataprocess = dp()

for i in range(200):
    name = Source.Native.INHALE_DIR + str(i) + ".wav"
    try:
        dataprocess.prepare_data(name)
    except:
        print("oops... check file: ", name)

for i in range(400):
    name = Source.Native.UNKNOWN_DIR + str(i) + ".wav"
    try:
        dataprocess.prepare_data(name)
    except:
        print("oops... check file: ", name)

for i in range(200):
    name = Source.Native.EXHALE_DIR + str(i) + ".wav"
    try:
        dataprocess.prepare_data(name)
    except:
        print("oops... check file: ", name)



print("all good!")