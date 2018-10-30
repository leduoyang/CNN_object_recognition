import sys
import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.examples.tutorials.mnist import input_data
import csv

def dataPreProcess(path):
    df = pd.DataFrame(columns={"fname","path"})
    n=0
    for filename in sorted(os.listdir(path)):
        df.loc[n] = pd.Series({"fname":filename[0:4],"path":path+filename})
        n=n+1
    x_test = readImgs(df["path"].values)
    return x_test,df["fname"].values 

def readImgs(file_paths):
    h , w, ch= cv2.imread(file_paths[0]).shape
    X = np.zeros([len(file_paths),h,w,1])
    count = 0
    for path in file_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X[count,:,:,0] = img
        count = count + 1
    return X

def buildModel():
    num_classes = 10
    # LeNet-5 : input => con => pooling => con => pooling => full connection => output
    INPUT = Input(shape = (28,28,1))
    CONV1 = Conv2D(10,(5,5),padding='valid',activation='relu',name='CONV1')(INPUT)
    MP1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(CONV1)
    DO1 = Dropout(0.2, name='DO1')(MP1)
    CONV2 = Conv2D(10,(5,5),padding='valid',activation='relu',name='CONV2')(DO1)
    MP2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(CONV2)
    DO2 = Dropout(0.2, name='DO2')(MP2)
    FLAT = Flatten()(DO2)
    DEN1 = Dense(500,activation='relu',name='DEN1')(FLAT)
    FC1 = Dense(num_classes,activation='softmax',name='FC1')(DEN1)
    
    model = Model(INPUT,FC1)
    h_model = Model(INPUT,CONV1)
    low_model = Model(INPUT,DO1)
    
    return model , h_model , low_model

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2 :
        raise Exception('please enter paths of the data and output file')    
    
    tData_path = argv[1]
    outputF_path = argv[2]
    
    model_name = "CNN_model_e20"
    model , h_model , low_model = buildModel()
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-4),metrics=['accuracy'])
    model.load_weights(model_name)    
    model.summary()
    
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    test_img = mnist.test.images
    l , d = test_img.shape
    test_img = test_img.reshape((l,28,28,1))
    test_label = mnist.test.labels
    acc = model.evaluate(test_img,test_label,verbose=1)
    print('Test accuracy :'+str(acc))
    
    
    x_test,fnames = dataPreProcess(tData_path)
    prediction = model.predict(x_test)
    labels = np.argmax(prediction, axis=1)
    
    
    text = open(outputF_path , "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id","value"])
    for i in range(len(labels)):
        s.writerow([fnames[i],labels[i]])
    text.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    