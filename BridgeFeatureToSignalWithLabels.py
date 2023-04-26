from DataSets.SignaksWithLabels import *
from features.FeatureExtractor import *


Signals, Labels, fs = GetSigalFromDataset()
ecg_features = Features(feature_groups=['full_waveform_features'])
Features_List = []
Label_List = []
for i in range(len(Signals)):
    # if i == 2:
    #     break
    # Feature_List=[]
    # print(len(Features_List))
    rawSignal = []
    threshHold=700
    label = "N"
    for j in range(len(Signals[i])):
        print("i = ", i, ", whole_i = ", len(Signals), ", j = ", j, ", whole_j = ", len(Signals[i]))
        if Labels[i][j]!="N":
            label = "abN"
        for c in range(len(Signals[i][j])):
            rawSignal.append(Signals[i][j][c])
            if len(rawSignal)==threshHold:
                # print(len(rawSignal))
                ts, signal_raw, signal_filtered, rpeaks, templates_ts, templates = ecg_features._preprocess_signal(
                    fs=fs[i],
                    signal_raw=rawSignal, filter_bandwidth=[3, 45], normalize=True,
                    polarity_check=True, template_before=0.2, template_after=0.4
                )
                try:
                    if ts == -1:
                        threshHold+=100
                        continue
                except:
                    pass
                features = ecg_features._group_features(ts=ts, signal_raw=signal_raw,
                                        signal_filtered=signal_filtered, rpeaks=rpeaks,
                                        templates_ts=templates_ts, templates=templates)
                Features_List.append(features)
                Label_List.append(label)
                # print(label)
                threshHold=700
                rawSignal=[]
                label = "N"
np.save("X_train_Features.npy", np.array(Features_List))
np.save("Y_train_Features.npy", np.array(Label_List))