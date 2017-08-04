import numpy as np
import scipy.io.wavfile

import os
import argparse 

import numpy as np
import scipy.io.wavfile as wav 
import speechUtil as spcutl
from sklearn import svm
from sklearn.decomposition import KernelPCA
import librosa

trainFolder='trainingData'
testFolder='testingData'



y= [1,10,11,2,3,4,5,6,7,8,9]

clf = svm.SVC( C=1.0, kernel='rbf', degree=3,
                    gamma='auto', coef0=0.0, shrinking=True, 
                    probability=False, tol=0.001, cache_size=200,
                    class_weight=None, verbose=False, max_iter=-1, 
                    decision_function_shape=None, random_state=None)

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True)

def kernel_pca(X):
    return kpca.fit_transform(X)
    

def get_speech_data(folder):
    X=[]    
    for filename in [x for x in os.listdir(folder) if x.endswith('.wav')]:            
        filepath = os.path.join(folder, filename)
        print('file',filepath)        
        y, sr = librosa.load(filepath)               
        mfcc_features = librosa.feature.mfcc(y,sr)
        kpca_mfcc_scalar=np.mean(mfcc_features, axis=1)
        print(kpca_mfcc_scalar)
#        
#         kpca_mfcc_scalar=mfcc_features.flatten()
#        
        #print('kpca_mfcc_scalar shape',kpca_mfcc_scalar.shape)
        X.append(kpca_mfcc_scalar)
    return X

def train_test_model():
    X_train=get_speech_data(trainFolder)
    print('xtrain ',X_train)
    #clf.fit(X_train, y)
    X_test=get_speech_data(testFolder)
    #print(clf.predict(X_test))
     
train_test_model()


