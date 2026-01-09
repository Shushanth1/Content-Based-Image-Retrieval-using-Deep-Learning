from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
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
from keras.applications import MobileNet
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
import webbrowser
import json

main = tkinter.Tk()
main.title("Content Based Image Retrieval using Deep Learning")
main.geometry("1300x1200")

global filename
global X, Y, X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore
global vgg_classifier
labels = []

def getLabel(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def upload():
    global filename, labels
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir = ".")
    pathlabel.config(text=filename)
    text.insert(END,"Dataset loaded\n\n")
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name)
    text.insert(END,"Available Images class labels found in dataset: "+str(labels))

def imageProcessing():
    global filename
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        labels.clear()
        path = filename
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
    X = X.astype('float32')
    X = X/255
    test = X[320]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)        
    text.insert(END,"Total images found in dataset: "+str(X.shape[0])+"\n")
    text.insert(END,"Dataset train & test split details\n")
    text.insert(END,"80% images used for training & 20% images used for testing\n")
    text.insert(END,"Total images used to train Deep Learning Algorithms: "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total images used to test Deep Learning Algorithms: "+str(X_test.shape[0])+"\n")
    cv2.imshow("Processed Sample Image",cv2.resize(test,(200,200)))
    cv2.waitKey(0)

#testprediction function to calculate metrics
def testPrediction(name, classifier, X_test, y_test):
    global labels
    predict = classifier.predict(X_test) #perform prediction using VGG, Resnet or mobilenet classifier
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    p = precision_score(testY, predict,average='macro') * 100 #calculate precision and other metrics
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    text.insert(END,"Test Images Size: "+str(X_test.shape[0])+"\n")
    text.insert(END,name+' Accuracy  : '+str(a)+"\n")
    text.insert(END,name+' Precision : '+str(p)+"\n")
    text.insert(END,name+' Recall    : '+str(r)+"\n")
    text.insert(END,name+' FMeasure  : '+str(f)+"\n")
    conf_matrix = confusion_matrix(testY, predict)
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    FN = 180 - (TP + FP)
    TN = 180 - (TP + FP)
    text.insert(END,name+" TP = "+str(np.sum(TP) / X_test.shape[0])+" FP = "+str(np.sum(FP) / X_test.shape[0])+" TN = 0 FN = 0\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    plt.title(name+" Confusion matrix") 
    plt.show()
    

def runResnet():
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore, jaccard, error_rate, auc
    accuracy = []
    precision = []
    recall = []
    fscore = []
    jaccard = []
    error_rate = []
    auc = []
    if os.path.exists('model/resnet_model.json'):
        with open('model/resnet_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()
        classifier.load_weights("model/resnet_model_weights.h5")
        classifier._make_predict_function()       
    else:
        #defining RESNET MODEL
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
    print(str(X_test.shape)+" "+str(y_test.shape))
    testPrediction("Resnet Algorithm", classifier, X_test, y_test)
    
def runVGG():
    global vgg_classifier
    if os.path.exists('model/vgg_model.json'):
        with open('model/vgg_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()    
        classifier.load_weights("model/vgg_model_weights.h5")
        classifier._make_predict_function()
        classifier.layers.pop()
        layer = classifier.layers[-2]
        vgg_classifier = Model(inputs=classifier.input, outputs=layer.output)
    else:
        #defining VGG model
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
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X, Y, batch_size=16, epochs=20, shuffle=True, verbose=2, validation_data=(X_test, y_test))
        classifier.save_weights('model/vgg_model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/vgg_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/vgg_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(classifier.summary())
    print(str(X_test.shape)+" "+str(y_test.shape))
    testPrediction("VGG16 Algorithm", classifier, X_test, y_test)

def runMobilenet():
    if os.path.exists('model/mobilenet_model.json'):
        with open('model/mobilenet_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()    
        classifier.load_weights("model/mobilenet_model_weights.h5")
        classifier._make_predict_function()       
    else:
        #defining mobilenet object
        mobilenet = MobileNet(input_shape = (32, 32, 3), include_top = False, weights = "imagenet")
        #last google net model layer will be ignore to concatenate banana custom model
        inception.trainable = False
        classifier = Sequential()
        #adding mobilenet features to our model
        classifier.add(mobilenet)
        #defining CNN layer with 32 filters to filter image features
        classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
        #using max pooling we will collect all filter features
        classifier.add(MaxPooling2D(pool_size = (1,1)))
        classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (1, 1)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
        #compiling the model
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        #start training model
        hist = classifier.fit(X, Y, batch_size=16, epochs=20, shuffle=True, verbose=2, validation_data=(X_test, y_test))
        classifier.save_weights('model/mobilenet_model_weights.h5')#save the weights for future used            
        model_json = classifier.to_json()
        with open("model/mobilenet_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/mobilenet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(classifier.summary())
    #calling testPrediction function to perform prediction using mobilenet on test data and then calculate accuracy and other metrics
    testPrediction("MobileNet Algorithm", classifier, X_test, y_test)

def graph():
    output = "<table align=center border=1><tr><th>Algorithm Name</th><th>Precision</th><th>Recall</th><th>F1 SCORE</th><th>Accuracy</th></tr>"
    columns = ["Algorithm Name","Precison","Recall","FScore","Accuracy"]
    values = []
    algorithm_names = ["Resnet","VGG16","MobileNet"]
    for i in range(len(algorithm_names)):
        output += "<tr><td>"+algorithm_names[i]+"</td>"
        output += "<td>"+str(precision[i])+"</td>"
        output += "<td>"+str(recall[i])+"</td>"
        output += "<td>"+str(fscore[i])+"</td>"
        output += "<td>"+str(accuracy[i])+"</td></tr>"        
    f = open("output.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("output.html",new=1)       
    df = pd.DataFrame([['Resnet','Precision',precision[0]],['Resnet','Recall',recall[0]],['Resnet','F1 Score',fscore[0]],['Resnet','Accuracy',accuracy[0]],
                       ['VGG16','Precision',precision[1]],['VGG16','Recall',recall[1]],['VGG16','F1 Score',fscore[1]],['VGG16','Accuracy',accuracy[1]],
                       ['MobileNet','Precision',precision[2]],['MobileNet','Recall',recall[2]],['MobileNet','F1 Score',fscore[2]],['MobileNet','Accuracy',accuracy[2]],
                        
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("Resnet, VGG16, & MobileNet Performance Graph")
    plt.show()

def close():
    main.destroy()

def format_image_for_display(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (300, 300))
    return img

def read_and_format_image(img_path):
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, (32, 32))
    return resized_img.reshape((1, 32, 32, 3))


def cosineBasedimageRetrieval(query, vgg_classifier): #cosine based image similarity calculation 
    dataset = json.load(open('feat_vectors.json'))
    distances = []
    for key in list(dataset.keys()):
        dico = {}
        img = dataset[key]
        cosine_similarity = np.linalg.norm(np.array(query) - np.array(img))
        print(cosine_similarity)
        if cosine_similarity < 5:
            dico['img_name'] = key
            dico['distance'] = cosine_similarity
            distances.append(dico)
    distances.sort(key=lambda x: x['distance'])
    images = [d['img_name'] for d in distances]
    list_images = []
    for i in range(len(images)):
        print(images[i])
        list_images.append(format_image_for_display(images[i]))
    return list_images 

def cbir():
    text.delete('1.0', END)
    global vgg_classifier
    filename = askopenfilename(initialdir = "queryImages") #uploading query image
    pathlabel.config(text=filename)
    text.insert(END,filename+" loaded\n")
    input_img = read_and_format_image(filename) #reading image from given path
    query = list(vgg_classifier.predict(input_img / 255).flatten().astype(float))#using VGG16 we are predicting or extracting features for given input image
    list_images  = cosineBasedimageRetrieval(query, vgg_classifier)#now calling cosine base function to calculate similariy between images to perform retrieval
    temp = []
    for i in range(len(list_images)):
        temp.append(list_images[i])
        if i > 10:
            break
    list1 = []
    list2 = []
    list3 = []
    list1.append(temp[0])
    list1.append(temp[1])
    list1.append(temp[2])    
    list2.append(temp[3])
    list2.append(temp[4])
    list2.append(temp[5])
    list3.append(temp[6])
    list3.append(temp[7])
    list3.append(temp[8])
    list3.append(temp[9])  
    stack1 = cv2.hconcat(list1)
    stack2 = cv2.hconcat(list2)
    stack3 = cv2.hconcat(list3)
    query = format_image_for_display(filename)
    cv2.imshow("Original Image",query)
    cv2.imshow('Retrieved images 1 to 3', stack1)
    cv2.imshow('Retrieved images 4 to 6', stack2)
    cv2.imshow('Retrieved images 6 to 10', stack3)
    cv2.waitKey(0)    


def close():
    main.destroy()
    
font = ('times', 16, 'bold')
title = Label(main, text='Content Based Image Retrieval using Deep Learning')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Coral Dataset", command=upload)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=350,y=100)

preprocess = Button(main, text="Image Preprocessing", command=imageProcessing)
preprocess.place(x=50,y=150)
preprocess.config(font=font1) 

resnetmodel = Button(main, text="Train Resnet Algorithm", command=runResnet)
resnetmodel.place(x=300,y=150)
resnetmodel.config(font=font1) 

vggmodel = Button(main, text="Train VGG16 Algorithm", command=runVGG)
vggmodel.place(x=550,y=150)
vggmodel.config(font=font1) 

mnmodel = Button(main, text="Train MobileNet Algorithm", command=runMobilenet)
mnmodel.place(x=810,y=150)
mnmodel.config(font=font1) 

graphButton = Button(main, text="All Algorithms Performance Graph", command=graph)
graphButton.place(x=50,y=200)
graphButton.config(font=font1)

cbirButton = Button(main, text="CBIR using Test Image", command=cbir)
cbirButton.place(x=380,y=200)
cbirButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=810,y=200)
exitButton.config(font=font1) 

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
