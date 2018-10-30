import sys
import os
import pdb
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from sklearn.manifold import TSNE
from keras import backend as K
 
def getFileinfo(dir_path):
    df = pd.DataFrame(columns={"action","labels","paths"})
    count = 0
    for sub_dir in tqdm(os.listdir(dir_path)):
        for label in os.listdir(dir_path + '/' +sub_dir):
            for file in os.listdir(dir_path + sub_dir + '/' + label):
                filename = dir_path + sub_dir + '/' + label + '/' + file
                df.loc[count] = pd.Series({'action':sub_dir, 'labels':label[len(label)-1], 'paths':filename})
                count = count + 1
    return df

def dataPreProcess(files_info):
    train_data , valid_data = files_info[ files_info["action"] == "train" ] , files_info[ files_info["action"] == "valid" ]
    
    print('read training data...')
    train_x , train_y = readImgs( train_data["paths"].values ) , to_categorical(np.array(train_data["labels"].values))
    
    print('read validation data...')
    valid_x , valid_y = readImgs( valid_data["paths"].values ) , to_categorical(np.array(valid_data["labels"].values))
    
    return train_x , train_y , valid_x , valid_y

def readImgs(file_paths):
    h , w, ch= cv2.imread(file_paths[0]).shape
    X = np.zeros([len(file_paths),h,w,1])
    count = 0
    for path in file_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = normalization(img)
        X[count,:,:,0] = img
        count = count + 1
    return X
def normalization(img):
    mean = np.mean(img)
    #pdb.set_trace()
    std = np.std(img)
    img = (img - mean)/std
    #pdb.set_trace()
    return img

def valid_data_for_tsne(valid_x,valid_y):
    n , h , w , ch= valid_x.shape
    tsne_data = np.zeros([1000,h,w,1])
    tsne_label = np.zeros([1000,])
    num_classes = 10
    for i in range(num_classes):
        tsne_data[ 100 * i : 100 * (i+1)] = valid_x[1000 * i : 1000 * i + 100]
        tsne_label[100 * i : 100 * (i+1)] = np.argmax(valid_y[1000 * i : 1000 * i + 100],axis=1)
        #pdb.set_trace()
    return tsne_data,tsne_label
    
def buildModel():
    num_classes = 10
    # LeNet-5 : input => con => pooling => con => pooling => full connection => output
    INPUT = Input(shape = (28,28,1))
    CONV1 = Conv2D(30,(5,5),padding='valid',activation='relu',name='CONV1')(INPUT)
    MP1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(CONV1)
    DO1 = Dropout(0.2, name='DO1')(MP1)
    CONV2 = Conv2D(30,(5,5),padding='valid',activation='relu',name='CONV2')(DO1)
    MP2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(CONV2)
    DO2 = Dropout(0.2, name='DO2')(MP2)
    FLAT = Flatten()(DO2)
    DEN1 = Dense(500,activation='relu',name='DEN1')(FLAT)
    FC1 = Dense(num_classes,activation='softmax',name='FC1')(DEN1)
    
    model = Model(INPUT,FC1)
    h_model = Model(INPUT,CONV2)
    l_model = Model(INPUT,CONV1)
    
    return model , h_model , l_model
    
if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2:
        raise Exception('please enter filename')
    folder_path = argv[1]
    #files_info = getFileinfo(folder_path)
    #files_info.to_csv('files_info')
    
    files_info = pd.read_csv('files_info')
    train_x , train_y , valid_x , valid_y = dataPreProcess(files_info)
    np.save("train_x", train_x)
    np.save("train_y", train_y)
    np.save("valid_x", valid_x)
    np.save("valid_y", valid_y)
    
    #train_x=np.load("train_x.npy")
    #train_y=np.load("train_y.npy")
    #valid_x=np.load("valid_x.npy")
    #valid_y=np.load("valid_y.npy")    
    ########################################################################### 
    # Build a CNN model and train it on the given dataset. Show the architecture of your model in the report.
    model , h_model , l_model = buildModel()
    model.summary()
    
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-4),metrics=['accuracy'])
    ckpt = ModelCheckpoint('CNN_model_e{epoch:02d}',monitor='val_acc',save_best_only=False,save_weights_only=True,verbose=1)
    cb= [ckpt]
    
    
    epochs = 20
    batch_size = 256
    history = model.fit(train_x,train_y,
                        batch_size=batch_size,
                        epochs=epochs,validation_data=(valid_x,valid_y),callbacks=cb,verbose=1,shuffle=True)
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(np.arange(epochs),loss,'b',label='train loss')
    plt.plot(np.arange(epochs),val_loss,'r',label='valid loss')    
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("loss curve")
    plt.legend(loc='best')
    
    plt.subplot(122)
    plt.plot(np.arange(epochs)+1,acc,'b',label='train accuracy')
    plt.plot(np.arange(epochs)+1,val_acc,'r',label='valid accuracy')    
    plt.xlabel("epochs")
    plt.ylabel("accuarcy")
    plt.title("accuarcy curve")  
    plt.legend(loc='best')
    plt.tight_layout()
    
    plt.show()
    
    ###########################################################################
    # Visualize high-level and low-level features of 1000 validation data
    tsne_data,tsne_label = valid_data_for_tsne(valid_x,valid_y)
    dh = h_model.predict(tsne_data)
    n,h,w,ch=dh.shape
    dh = dh.reshape((n,h*w*ch))
    dl = l_model.predict(tsne_data)  
    n,h,w,ch=dl.shape
    dl = dl.reshape((n,h*w*ch))
    dh_embedded = TSNE(n_components=2).fit_transform(dh)
    dl_embedded = TSNE(n_components=2).fit_transform(dl)
    
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.scatter(dl_embedded[:,0], dl_embedded[:,1], c=tsne_label) 
    plt.title("low-level features")
    
    plt.subplot(122)
    plt.scatter(dh_embedded[:,0], dh_embedded[:,1], c=tsne_label)   
    plt.title("high-level features")
    plt.show()    
    
    
    ###########################################################################
    # Visualize at least 6 filters on both the first and the last convolutional layers.
    
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    input_img = model.layers[0].input
    layer_name = 'CONV2'
    
    for filter_index in range(6):
        #filter_index = range(7)  # can be any integer from 0 to 511, as there are 512 filters in that layer
    
    
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        loss = K.sum(layer_output[:, :, :, filter_index])
        
        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]
        
        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        
        # this function returns the loss and grads given the input picture
        iterate1 = K.function([input_img], [loss])
        iterate2 = K.function([input_img,loss], [grads])
        
        iterate = K.function([input_img], [loss,grads])
        
        # we start from a gray image with some noise
        input_img_data = np.random.random((1, 28, 28,1)) * 20 + 128.
        
        step = 50
        tem1=0
        tem2=0
        # run gradient ascent for 20 steps
        for i in range(20):
            #pdb.set_trace()
            #loss = iterate1([input_img_data])
            #grads_value = iterate2([input_img_data,loss])
            #tem1 = tem1 + sum(sum(sum(sum(grads_value))))
            loss,grads_value = iterate([input_img_data])
            #tem2 = tem2 + sum(sum(sum(sum(grads_value))))
            input_img_data = input_img_data +np.dot(step,grads_value)
            #input_img_data = input_img_data[0,:,:,:,:]
            #pdb.set_trace()
    
        
        plt.figure(filter_index)
        img = input_img_data[0]
        img =img.reshape((28,28))
        plt.imshow(img)
    
    
    
    
    
    
    
    
    
    
    