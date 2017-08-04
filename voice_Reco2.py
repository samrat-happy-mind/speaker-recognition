import os
import argparse 

import numpy as np
import scipy.io.wavfile as wav 
import speechUtil as spcutl
from sklearn import svm
from sklearn.decomposition import KernelPCA

trainFolder='trainingData'
testFolder='testingData'

############# Extract MFCC features #############


y= [0,0,1,1,1,2,2,2,3,3,3]

clf = svm.SVC( C=1.0, kernel='rbf', degree=3,
                    gamma='auto', coef0=0.0, shrinking=True, 
                    probability=False, tol=0.001, cache_size=200,
                    class_weight=None, verbose=False, max_iter=-1, 
                    decision_function_shape=None, random_state=None)

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True)


def kernel_pca(X):
    return kpca.fit_transform(X)
    

def get_speech_data(folder,X):
    
    for filename in [x for x in os.listdir(folder) if x.endswith('.wav')]:
                # Read the input file
        filepath = os.path.join(folder, filename)
        sampling_freq, audio = wav.read(filepath)
        # Extract MFCC features
        mfcc_features = spcutl.mfcc(audio,sampling_freq)
        
        kpca_mfcc=kernel_pca(mfcc_features)
        print('kpca_mfccc shape',kpca_mfcc.shape)
        print('kpca_mfcc ',kpca_mfcc)
        # Append to the variable X
        if len(X) == 0:
            X = mfcc_features
        else:
            X = np.append(X,mfcc_features,axis=0)       
    return X

def train_test_model():
    X_train=get_speech_data(trainFolder,np.array([]))
    print('xtrain ',X_train)
    print('xtrain_shape',X_train.shape)
    #clf.fit(X_train, y)
    #X_test=get_speech_data(testFolder,np.array([]))
    #clf.predict(X_test)
     
train_test_model()
