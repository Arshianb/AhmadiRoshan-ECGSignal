import wfdb
import numpy as np
import matplotlib.pyplot as plt
import sys
# sys.path.append("/home/arshianb/Desktop/طرح احمدی روشن/Code/")
# from MyFeatures.FeatureExtractor1 import *
import glob
def GetSigalFromDataset(SorcePath = "/home/arshianb/Desktop/طرح احمدی روشن/Code/DataSets/mit-bih-arrhythmia-database-1.0.0"):
    Signals = []
    Labels = []
    fs = []
    for file_name in glob.glob("{}/*.dat".format(SorcePath)):
        Signal = []
        Label = []
        # Load ECG signal
        index = file_name[file_name.find(SorcePath)+len(SorcePath)+1:file_name.find(".dat")]
        # print(index)
        record = wfdb.rdrecord("{}/{}".format(SorcePath, index))
        # signal = record.p_signal.flatten()
        signal = list(record.p_signal[:, 0])
        ann = wfdb.rdann("{}/{}".format(SorcePath, index), 'atr')
        # print(ann.sample[0])
        beat_labels = list(ann.symbol)
        ann.sample = list(ann.sample)
        # ann.sample.insert(0, 0)
        # beat_labels.pop(0)
        # ann.sample[0] = 0
        # print(len(beat_labels))
        # print(len(ann.sample))
        # for i in range(0, len(ann.sample), 6):
        #     fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
        #     for j, ax in enumerate(axs.flatten()):
        #         ax.plot(range(len(signal[ann.sample[j+i]:ann.sample[j+i+1]])), signal[ann.sample[j+i]:ann.sample[j+i+1]])
        #         ax.set_title("item {} is {}".format(i+j, beat_labels[j+i]))
        #     plt.tight_layout()
        #     plt.savefig("/home/arshianb/Desktop/طرح احمدی روشن/Code/DataSets/plot/{}.png".format(i))
        #     plt.show()
        #     # plt.close()
        # print(file_name)
        for i in range(len(beat_labels)):
            # print(i)
            # if i+1 != len(ann.sample):
            #     Signals.append(signal[ann.sample[i]:ann.sample[i+1]])
            # else:
            #     Signals.append(signal[ann.sample[i]:])
            if i == 0:
                Signal.append(signal[0:ann.sample[i]])
            else:
                Signal.append(signal[ann.sample[i-1]:ann.sample[i]])
            Label.append(beat_labels[i])
        Signals.append(Signal)
        Labels.append(Label)
        fs.append(ann.fs)
            # print(len(Signals[-1]))
        # exit()
    return Signals, Labels, fs
