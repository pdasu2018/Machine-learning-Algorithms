import scipy.io
import numpy as np
from sklearn.preprocessing import StandardScaler

samples = scipy.io.loadmat('knn_data.mat')

traind = samples['train_data']
testd = samples['test_data']

standx = StandardScaler().fit_transform(traind)
standx2 = StandardScaler().fit_transform(testd)

mn = np.mean(standx, axis=0)
mn2 = np.mean(standx2, axis=0)
cov1= (standx - mn).T.dot((standx - mn)) / (standx.shape[0]-1)
cov2 = (standx2 - mn2).T.dot((standx2 - mn2)) / (standx2.shape[0]-1)

cov1= np.cov(standx.T)
cov2 = np.cov(standx2.T)

eTrainValue, eTrainVector = np.linalg.eig(cov1)
eTestValue, eTestVector = np.linalg.eig(cov2)

eTrain = [(np.abs(eTrainValue[i]), eTrainVector[:,i]) for i in range(len(eTrainValue))]
eTrain = [(np.abs(eTestValue[i]), eTestVector[:,i]) for i in range(len(eTestValue))]


eTrain.sort()
eTrain.reverse()
eTrain = eTrain[:50]

eTrain.sort()
eTrain.reverse()
eTrain = eTrain[:50]
W1 = np.hstack((eTrain[0][1].reshape(166,1),
                      eTrain[1][1].reshape(166,1), eTrain[2][1].reshape(166,1),
                     eTrain[3][1].reshape(166,1), eTrain[4][1].reshape(166,1),
                     eTrain[5][1].reshape(166,1), eTrain[6][1].reshape(166,1),
                     eTrain[7][1].reshape(166,1), eTrain[8][1].reshape(166,1),
                     eTrain[9][1].reshape(166,1), eTrain[10][1].reshape(166,1),
                     eTrain[11][1].reshape(166,1), eTrain[12][1].reshape(166,1),
                     eTrain[13][1].reshape(166,1), eTrain[14][1].reshape(166,1),
                     eTrain[15][1].reshape(166,1), eTrain[16][1].reshape(166,1),
                     eTrain[17][1].reshape(166,1), eTrain[18][1].reshape(166,1),
                     eTrain[19][1].reshape(166,1), eTrain[20][1].reshape(166,1),
                     eTrain[21][1].reshape(166,1), eTrain[22][1].reshape(166,1),
                     eTrain[23][1].reshape(166,1), eTrain[24][1].reshape(166,1),
                     eTrain[25][1].reshape(166,1), eTrain[26][1].reshape(166,1),
                     eTrain[27][1].reshape(166,1), eTrain[28][1].reshape(166,1),
                     eTrain[29][1].reshape(166,1), eTrain[30][1].reshape(166,1),
                     eTrain[31][1].reshape(166,1), eTrain[32][1].reshape(166,1),
                     eTrain[33][1].reshape(166,1), eTrain[34][1].reshape(166,1),
                     eTrain[35][1].reshape(166,1), eTrain[36][1].reshape(166,1),
                     eTrain[37][1].reshape(166,1), eTrain[38][1].reshape(166,1),
                     eTrain[39][1].reshape(166,1), eTrain[40][1].reshape(166,1),
                     eTrain[41][1].reshape(166,1), eTrain[42][1].reshape(166,1),
                     eTrain[43][1].reshape(166,1), eTrain[44][1].reshape(166,1),
                     eTrain[45][1].reshape(166,1), eTrain[46][1].reshape(166,1),
                     eTrain[47][1].reshape(166,1), eTrain[48][1].reshape(166,1),
                     eTrain[49][1].reshape(166,1)))

W2 = np.hstack((eTrain[0][1].reshape(166,1),
                      eTrain[1][1].reshape(166,1), eTrain[2][1].reshape(166,1),
                     eTrain[3][1].reshape(166,1), eTrain[4][1].reshape(166,1),
                     eTrain[5][1].reshape(166,1), eTrain[6][1].reshape(166,1),
                     eTrain[7][1].reshape(166,1), eTrain[8][1].reshape(166,1),
                     eTrain[9][1].reshape(166,1), eTrain[10][1].reshape(166,1),
                     eTrain[11][1].reshape(166,1), eTrain[12][1].reshape(166,1),
                     eTrain[13][1].reshape(166,1), eTrain[14][1].reshape(166,1),
                     eTrain[15][1].reshape(166,1), eTrain[16][1].reshape(166,1),
                     eTrain[17][1].reshape(166,1), eTrain[18][1].reshape(166,1),
                     eTrain[19][1].reshape(166,1), eTrain[20][1].reshape(166,1),
                     eTrain[21][1].reshape(166,1), eTrain[22][1].reshape(166,1),
                     eTrain[23][1].reshape(166,1), eTrain[24][1].reshape(166,1),
                     eTrain[25][1].reshape(166,1), eTrain[26][1].reshape(166,1),
                     eTrain[27][1].reshape(166,1), eTrain[28][1].reshape(166,1),
                     eTrain[29][1].reshape(166,1), eTrain[30][1].reshape(166,1),
                     eTrain[31][1].reshape(166,1), eTrain[32][1].reshape(166,1),
                     eTrain[33][1].reshape(166,1), eTrain[34][1].reshape(166,1),
                     eTrain[35][1].reshape(166,1), eTrain[36][1].reshape(166,1),
                     eTrain[37][1].reshape(166,1), eTrain[38][1].reshape(166,1),
                     eTrain[39][1].reshape(166,1), eTrain[40][1].reshape(166,1), 
                     eTrain[41][1].reshape(166,1), eTrain[42][1].reshape(166,1),
                     eTrain[43][1].reshape(166,1), eTrain[44][1].reshape(166,1),
                     eTrain[45][1].reshape(166,1), eTrain[46][1].reshape(166,1),
                     eTrain[47][1].reshape(166,1), eTrain[48][1].reshape(166,1),
                     eTrain[49][1].reshape(166,1)))

tn = standx.dot(W1)
tt = standx2.dot(W2)

np.savetxt("finaltrain.csv", tn, delimiter=",")
np.savetxt("finaltest.csv", tt, delimiter=",")
