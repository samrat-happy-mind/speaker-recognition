
import os

from sklearn.decomposition import KernelPCA
import librosa
from keras.models import Sequential
from keras.layers.core import Dense,Dropout
import numpy as np
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True)


def baseline_model(input_dim, nb_classes):    
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def kernel_pca(X):
    return kpca.fit_transform(X)
    

def get_speech_data(folder):
    X=[]
    for filename in [x for x in os.listdir(folder) if x.endswith('.wav')]:            
        filepath = os.path.join(folder, filename)                    
        y, sr = librosa.load(filepath)               
        mfcc_features = librosa.feature.mfcc(y,sr)
        kpca_mfcc=kernel_pca(mfcc_features)
        kpca_mfcc_scalar=kpca_mfcc.flatten()        
        X.append(kpca_mfcc_scalar)
           
    return X

def train_test_model():
    X_train=np.array(get_speech_data('train'))
    Y_train=np_utils.to_categorical([1,10,11,2,3,4,5,6,7,8,9]) 
    
    nn_model= baseline_model(X_train.shape[1],Y_train.shape[1])
    nn_model.fit(X_train,Y_train,nb_epoch=20,verbose=1)
    
    print(X_train.shape)
    print(Y_train.shape)
    
    X_test=np.array(get_speech_data('test'))  
    
    print(X_test.shape)
   
    
    y_prob = nn_model.predict(X_test)
    y_classes = y_prob.argmax(axis=-1)
    
    print("score matching is :",y_classes)
     
train_test_model()


