

import numpy as np
import keras
import json
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
# from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add


model = load_model('model_new_19.h5')

model_temp = InceptionV3(weights='imagenet',input_shape=(299,299,3))

model_inception = Model(model_temp.input,model_temp.layers[-2].output)

def preprocess_img(img):
    img = image.load_img(img,target_size=(299,299))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    #Normalization
    img = preprocess_input(img)
    return img

def encode_img(img):
    img = preprocess_img(img)
    feature_vector = model_inception.predict(img)
    feature_vector = feature_vector.reshape((1,feature_vector.shape[1]))
    return feature_vector


with open('storage/word_to_idx.pkl','rb') as w2i:
    word_to_idx = pickle.load(w2i)

with open('storage/idx_to_word.pkl','rb') as i2w:
    idx_to_word = pickle.load(i2w)

def predict_caption(photo):
    
    in_text = "startseq"
    max_len=74
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #WOrd with max prob always - Greedy Sampling
        word = idx_to_word[ypred]
        in_text += (' ' + word)
        
        if word == "endseq":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

def caption_image(image):
    enc = encode_img(image)
    caption = predict_caption(enc)
    
    return caption
