import numpy as np

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [series[i:(i+window_size)] for i in range(len(series)-window_size)]
    y = [series[i+window_size] for i in range(len(series)-window_size)]
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    
    #layer 1: LSTM module with 5 hidden units
    model.add(LSTM(5,
            input_shape=(window_size, 1))) 
    #layer 2: fully connected layer with 1 unit
    model.add(Dense(1, activation = "linear"))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    test_text = text
    valid_english_characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '!', ',', '.', ':', ';', '?']
    for i in range(0,len(valid_english_characters)):
        #Leave in the test_text only characters that are not valid english characters.
        test_text = test_text.replace(valid_english_characters[i],'')
    #List all non-valid english characters.
    nonvalid_english_characters = list(set(test_text))
    #Print all non-valid english characters to check if there is any character that shouldn't be a  non-valid english character
    #print(nonvalid_english_characters)
    #Based on the characters analysis the following define the final non-valid english characters
    nonvalid_english_characters = ['/', '\\', '\f', '\v', '\n', '\t', '\r', '|', '~', '`', '(', '{', '[', '"', '&', '%', '*', ')', '}', ']', '+', '=', '-', '_', '<', '>','$', '@', '^', '#', "'", '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'è', 'é',  'à', 'â']
    #Remove non-valid english characters from text 
    for i in range(0,len(nonvalid_english_characters)):
        text = text.replace(nonvalid_english_characters[i],'')

    # shorten any extra dead space created above
    text = text.replace('  ',' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    cutoff = window_size
    
    while (len(text) > cutoff):
        inputs.append(text[cutoff - window_size : cutoff])
        outputs.append(text[cutoff])
        cutoff += step_size

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    #layer 1: LSTM with 200 hidden units
    model.add(LSTM(200, input_shape = (window_size, num_chars)))
    #layer 2 should be a linear module, fully connected, with len(chars) hidden units
    model.add(Dense(num_chars, activation='linear'))
    #layer 3 should be a softmax activation
    model.add(Activation('softmax'))
    model.summary()
    return model
