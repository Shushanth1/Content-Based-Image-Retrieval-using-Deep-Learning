import cv2
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os
from keras.models import Model
import json
import glob


def format_image_for_display(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (300, 300))
    return img

def read_and_format_image(img_path):
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, (32, 32))
    return resized_img.reshape((1, 32, 32, 3))

with open('model/vgg_model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    classifier = model_from_json(loaded_model_json)
json_file.close()
classifier.load_weights("model/vgg_model_weights.h5")
classifier._make_predict_function()
classifier.layers.pop()
layer = classifier.layers[-2]
model = Model(inputs=classifier.input, outputs=layer.output)
print(model.summary())
def compute_feat_vectors():
    feat_vectors = {}
    for root, dirs, directory in os.walk('CorelDataset'):
        for j in range(len(directory)):
            if 'Thumbs.db' not in directory[j]:
                print(root+"/"+directory[j])
                input = read_and_format_image(root+"/"+directory[j])
                prediction = list(model.predict(input / 255).flatten().astype(float))
                feat_vectors[root+"/"+directory[j]] = prediction
    # Saves the features
    with open('feat_vectors.json', 'w') as f:
        json.dump(feat_vectors, f)
    f.close()

def compute_closest_images(img_path):
    input = read_and_format_image(img_path)
    #model = get_model(224, 224, 3)
    query = list(model.predict(input / 255).flatten().astype(float))
    dataset = json.load(open('feat_vectors.json'))
    distances = []
    for key in list(dataset.keys()):
        dico = {}
        img = dataset[key]
        distance = np.linalg.norm(np.array(query) - np.array(img))
        print(distance)
        if distance < 5:
            dico['img_name'] = key
            dico['distance'] = distance
            distances.append(dico)
    distances.sort(key=lambda x: x['distance'])
    images = [d['img_name'] for d in distances]
    list_images = []

    for i in range(len(images)):
        print(images[i])
        list_images.append(format_image_for_display(images[i]))
    print(len(list_images))
    stack = cv2.hconcat(list_images)
    query = format_image_for_display(img_path)
    cv2.imshow("Original Image",query)
    cv2.imshow('Retrieved images', stack)
    cv2.waitKey(0)
    
#compute_feat_vectors()
compute_closest_images('CorelDataset/beaches/111.jpg')








    
