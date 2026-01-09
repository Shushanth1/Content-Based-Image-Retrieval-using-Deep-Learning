import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split 
from keras.applications import ResNet50
from keras.models import Model
from keras.applications import VGG16
from keras.applications import DenseNet121
from keras.applications import MobileNet
'''
def getLabel(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

labels = []
X = []
Y = []
path = 'CorelDataset'

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)
print(labels)

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (32,32))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(32,32,3)
            X.append(im2arr)
            label = getLabel(name)
            Y.append(label)
            print(str(j)+" "+name+" "+str(label))

X = np.asarray(X)
Y = np.asarray(Y)

np.save('model/X.txt',X)
np.save('model/Y.txt',Y)
'''
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')

X = X.astype('float32')
X = X/255
    
test = X[3]
cv2.imshow("aa",test)
cv2.waitKey(0)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
print(Y.shape)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
'''
if os.path.exists('model/resnet_model.json'):
    with open('model/resnet_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/resnet_model_weights.h5")
    classifier._make_predict_function()       
else:
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    for layer in resnet.layers:
        layer.trainable = False
    classifier = Sequential()
    classifier.add(resnet)
    classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X, Y, batch_size=16, epochs=20, shuffle=True, verbose=2, validation_data=(X_test, y_test))
    classifier.save_weights('model/resnet_model_weights.h5')            
    model_json = classifier.to_json()
    with open("model/resnet_model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/resnet_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
print(classifier.summary())
'''
'''
if os.path.exists('model/vgg_model.json'):
    with open('model/vgg_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/vgg_model_weights.h5")
    classifier._make_predict_function()       
else:
    vgg = VGG16(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights="imagenet")
    vgg.trainable = False
    classifier = Sequential()
    classifier.add(vgg)
    classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1,1)))
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X, Y, batch_size=16, epochs=20, shuffle=True, verbose=2, validation_data=(X_test, y_test))
    classifier.save_weights('model/vgg_model_weights.h5')            
    model_json = classifier.to_json()
    with open("model/vgg_model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/vgg_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
print(classifier.summary())
'''

if os.path.exists('model/mobilenet_model.json'):
    with open('model/mobilenet_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/mobilenet_model_weights.h5")
    classifier._make_predict_function()       
else:
    mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=(32,32,3))
    classifier = Sequential()
    classifier.add(mobilenet)
    classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1,1)))
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X, Y, batch_size=16, epochs=20, shuffle=True, verbose=2, validation_data=(X_test, y_test))
    classifier.save_weights('model/mobilenet_model_weights.h5')            
    model_json = classifier.to_json()
    with open("model/mobilenet_model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/mobilenet_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
print(classifier.summary())

