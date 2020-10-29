#Team Members
#Saivikas Meda - sxm190011
#Vijaya kaushika - vxa180028
#Sneha sam - ses190004
#Anushka - aum180000



from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import string
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
 
''' 
IMAGE FEATURES
Input - Directory containing images
1. Using VGG16 - Deep Convolutional Network 
2. Convert image to pixel vector (0-1)
Output - Dictionary - Key - Image name, Value - Feature vector'''

def extract_image_features(directory):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())
    features = dict()
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
        print('>%s' % name)
    return features
 
=directory = 'Flicker8k_Dataset'
features = extract_image_features(directory)
print("features ",features)
print('Extracted Features: %d' % len(features))


# Dump the features of images in an pickel file for future references
dump(features, open('image_features.pkl', 'wb'))






'''
LOAD DOCUMENT
Input - Filepath String to get the data from file
Output - Return the text data
'''
def load_document(filepath):
    filename = open(filepath, 'r')
    textdata = filename.read()
    filename.close()
    return textdata
 
filename = 'Flickr8k_text/Flickr8k.token.txt'
documentdata= load_document(filename)




'''
LOADING CAPTIONSET 
Input - Filepath string to get the captions
Output - Set of all image captions
'''

def load_descriptions(doc):
mapping = dict()
for line in doc.split('\n'):
    tokens = line.split()
    if len(line) < 2:
        continue
    image_id, image_desc = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    image_desc = ' '.join(image_desc)
    if image_id not in mapping:
        mapping[image_id] = list()
    mapping[image_id].append(image_desc)
return mapping

 

def load_captionset(filepath):
    documentcaptions = load_document(filepath)
    imagedataset = list()
    for caption in documentcaptions.split('\n'):
        if len(caption) < 1:
            continue
        identifier = caption.split('.')[0]
        imagedataset.append(identifier)
    return set(imagedataset)
 

'''
LOAD CLEAN DESCRIPTIONS FOR IMAGES 
Input - Filename as a string and dataset - a list of all image id's 
Output - A dictionary which contains image id as key and array of captions as value
'''

def load_descriptions_preproccessing(filename, dataset):
    doc = load_document(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions
 
def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] =  ' '.join(desc)
    
    
def to_vocabulary(descriptions):
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    
  
    
filename = 'Flickr8k_text/Flickr8k.token.txt'
doc = load_document(filename)
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
clean_descriptions(descriptions)
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
save_descriptions(descriptions, 'imagedescriptions.txt')





def load_image_features(filename, dataset):
	all_features = load(open(filename, 'rb'))
	image_features = {l: all_features[l] for l in dataset}
	return image_features
 
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_captionset(filename)
print('Dataset: %d' % len(train))
train_descriptions = load_descriptions_preproccessing('imagedescriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
train_features = load__features('image_features.pkl', train)
print('Photos: train=%d' % len(train_features))





'''
A LIST OF CLEAN IMAGE DESCRIPTIONS
Input - Dictionary with image id and captions 
Output - List of all captions
'''
def captionlines(desc):
    all_captions = list()
    for key in desc.keys():
        [all_captions.append(i) for i in desc[key]]
    return all_captions
 
'''
TOKENIZER FOR CAPTION DESCRIPTIONS
Input - Dictionary with image id and captions 
Output - Tokenizer for all descriptions
'''
def create_imagetokenizer(descriptions):
    lines = captionlines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# prepare tokenizer
tokenizer = create_imagetokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)



'''
MAX DESCRIPTION LENGTH
Input - Image Description  
Output - Maximum length of descriptions 
'''
def max_captionlength(descriptions):
    lines = captionlines(descriptions)
    return max(len(j.split()) for j in lines)
 
'''
CREATE SEQUENCES OF IMAGES, INPUT SEQUENCES AND OUTPUT WORDS FOR AN IMAGE
Input - Tokenizer for all descriptions, Maximum Length of caption description, Dictionary - Key - image_id, Value - a list of caption description, photos - image vector, vocab_size
Output - Padded input sequence, Encoded Output Sequence, Image vector with photos  
'''
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = list(), list(), list()
    for key, description_list in descriptions.items():
        for desc in description_list:
            sequence = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(sequence)):
                in_sequence, out_sequence = sequence[:i], sequence[i]
                in_sequence = pad_sequences([in_sequence], maxlen=max_length)[0]
                out_sequence = to_categorical([out_sequence], num_classes=vocab_size)[0]
                X1.append(photos[key][0])
                X2.append(in_sequence)
                y.append(out_sequence)
    return array(X1), array(X2), array(y)
 
'''
DEFINE THE CAPTIONING MODEL
Input - Number of words in document, Maximum length of description
Output - Trained Model 
'''
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model
 
 



'''
DATA GENERATOR 
Input - Descriptions - a list of all descriptions, photos - to retrieve image vector, tokenizer - word tokenizer, maximum description length, number of words 
Output - Padded input sequence, Encoded Output Sequence, Image vector with photos
'''
def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    while 1:
        for key, desc_list in descriptions.items():
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
            yield [[in_img, in_seq], out_word]



''' 
SEQUENCES OF IMAGES, INPUT SEQUENCES AND OUTPUT WORDS FOR AN IMAGE
Input - word tokenizer, maximum length of image description, list of all descriptions, image vector as value and image id as key, size of all words in the document
Output - image vector as value and image id as key, padding input sequence, categorical data of sequence 
'''
def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)


filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_captionset(filename)
print('Dataset: %d' % len(train))
train_descriptions = load_descriptions_preproccessing('imagedescriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
train_features = load_image_features('image_features.pkl', train)
print('Photos: train=%d' % len(train_features))
tokenizer = create_imagetokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
max_length = max_captionlength(train_descriptions)
print('Description Length: %d' % max_length)
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)
 




filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_captionset(filename)
print('Dataset: %d' % len(train))
train_descriptions = load_descriptions_preproccessing('imagedescriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
train_features = load_image_features('image_features.pkl', train)
print('Photos: train=%d' % len(train_features))
tokenizer = create_imagetokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
max_length = max_captionlength(train_descriptions)
print('Description Length: %d' % max_length)
 
model = define_model(vocab_size, max_length)
epochs = 3
steps = len(train_descriptions)
for i in range(epochs):
	generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
	model.save('model_' + str(i) + '.h5')



def id_to_word(integer, tokenizer):
    for word, i in tokenizer.word_index.items():
        if i == integer:
            return word
    return None

'''
GENERATE A DESCRIPTION FOR AN IMAGE
Input - Trained model, word tokenizer, photo vector and image id, maximum length of description
Output - generating the description for next iteration 
'''
def generate_description(model, tokenizer, photo, max_length):
    text = 'startseq'
    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yht = model.predict([photo,seq], verbose=0)
        yht = argmax(yht)
        word = id_to_word(yht, tokenizer)
        if word is None:
            break
        text += ' ' + word
        if word == 'endseq':
            break
    return in_text




'''
EVALUATING THE GENERATED MODEL
Input - Trained model, descriptions, photo dictionary with image id and vector, word tokenizer, maximum length of description
Output - BLEU score 
'''
def evaluate_testmodel(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    for key, description_list in descriptions.items():
        if key in photos.keys():
            yht = generate_description(model, tokenizer, photos[key], max_length)
        else:
            continue
        ref = [i.split() for i in description_list]
       
        actual.append(ref)
        predicted.append(yht.split())
    # calculate BLEU score
   
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(0.95, 0.01, 0.01, 0.03)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.25, 0.20, 0.05)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0.1)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))




from keras.models import load_model 
from nltk.translate.bleu_score import corpus_bleu


filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_captionset(filename)
print('Dataset: %d' % len(train))
train_desc = load_descriptions_preproccessing('imagedescriptions.txt', train)
print('Descriptions: train=%d' % len(train_desc))
tokenizer = create_imagetokenizer(train_desc)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
max_length = max_captionlength(train_desc)
print('Description Length: %d' % max_length)
 

filename = 'Flickr8k_text/Flickr_8k.testImages.txt'
test = load_captionset(filename)
print('Dataset: %d' % len(test))
test_desc = load_descriptions_preproccessing('imagedescriptions.txt', test)
print('Descriptions: test=%d' % len(test_desc))
test_features = load_image_features('image_features.pkl', test)
print('Photos: test=%d' % len(test_features))


filename = 'model_2.h5'
model = load_model(filename)
evaluate_testmodel(model, test_desc, test_features, tokenizer, max_length)




# Generate new caption
from keras.preprocessing.text import Tokenizer
from pickle import dump
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_captionset(filename)
print('Dataset: %d' % len(train))
train_desc = load_descriptions_preproccessing('imagedescriptions.txt', train)
print('Descriptions: train=%d' % len(train_desc))
tokenizer = create_imagetokenizer(train_desc)
dump(tokenizer, open('tokenizer.pkl', 'wb'))



tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length = 34
model = load_model('model_3.h5')



def extract_features(filename):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def id_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_description(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = argmax(yhat)
        word = id_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length = 34
model = load_model('model_3.h5')
photo = extract_features('Flicker8k_Dataset/3637013_c675de7705.jpg')
description = generate_description(model, tokenizer, photo, max_length)
print(description)
photo = extract_features('Flicker8k_Dataset/23445819_3a458716c1.jpg')
description = generate_description(model, tokenizer, photo, max_length)
print(description)
photo = extract_features('Flicker8k_Dataset/47871819_db55ac4699.jpg')
description = generate_description(model, tokenizer, photo, max_length)
print(description)
photo = extract_features('Flicker8k_Dataset/36422830_55c844bc2d.jpg')
description = generate_description(model, tokenizer, photo, max_length)
print(description)
