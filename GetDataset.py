import numpy as np

def GetData():
    XTemp = np.load("X_train_Features.npy", allow_pickle=True)
    Y = np.load("Y_train_Features.npy", allow_pickle=True)
    X=[]
    for i in XTemp:
        X.append(list(i[0].values()))
    X = np.array(X)
    # print(X.shape)
    return X, Y
    # print(len(X))
    # print(len(Y))
    # print("how_manyNormal = ", len(Y[np.where(Y=="N")]))
    # print("how_abNormal = ", len(Y[np.where(Y=="abN")]))
def find_max_min(Array):
    max_argumans = []
    min_argumans = []
    for i in range(Array.shape[1]):
        max_argumans.append(max(Array[:, i]))
        min_argumans.append(min(Array[:, i]))
    return max_argumans, min_argumans

if __name__=="__main__":
    X, Y = GetData()
    max_argumans, min_argumans = find_max_min(X)
    print(max_argumans)
    print(min_argumans)