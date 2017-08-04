import os
import pywt
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.decomposition import KernelPCA
from sklearn import svm
from sklearn import preprocessing
import numpy as np


kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True)
clf = svm.SVC( C=1.0, kernel='rbf', degree=3,
                gamma='auto', coef0=0.0, shrinking=True, 
                probability=False, tol=0.001, cache_size=200,
                class_weight=None, verbose=False, max_iter=-1, 
                decision_function_shape=None, random_state=None)

def getLabels():
    Y_svm=[0,0,1,1,1,2,2,2,3,3,3]
    return Y_svm
 
def waveletTransform(file_loc,file):
    filename=file_loc + '/' + file
    [fs,x] = wavfile.read(filename) 
    print(filename)
    cA, cD = pywt.dwt(x,'db1')     
    print('audio data',cD.transpose().shape)
    return cD.transpose()
        

def kernel_pca(X):
    return kpca.fit_transform(X)
      
def SVM_classify(X,y):    
    clf.fit(X, y) 
    return clf

def train_model():
    files=os.listdir('trainingData')
    pca_feed=np.concatenate([waveletTransform('trainingData',file) for file in files])         
    #X_kpca=kernel_pca(pca_feed)   
    Y_svm=getLabels()
    clf=SVM_classify(pca_feed,Y_svm)
    return clf
    
def test_model():
    clf=train_model()
    files=os.listdir('testingData')
    pca_feed=np.concatenate([waveletTransform('testingData',file) for file in files])
    print(pca_feed.shape)
    #X_kpca=kernel_pca(pca_feed)
    print(clf.predict(pca_feed))

test_model()




# kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
# 

# 
# X_kpca = kpca.fit_transform(cD)
# 
# print(X_kpca)



