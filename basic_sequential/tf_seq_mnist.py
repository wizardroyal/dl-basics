import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

def model_architecture(nn_archi, hl_activation="relu", ol_activation="softmax"):
    """
    Defines the architecture of an nn-model given the intended no of layers and nodes
    Arguments: 
        nn_archi: list of the number of nodes for each layer of the neural network
        hl_activation: the activation of the hidden layers
        ol_activation: the activation of the output layer
    Returns: a sequential nn-model, with architecture defined by nn-archi
    """
    L = len(nn_archi)  #no of layers including the input layer
    
    model = keras.Sequential()
    for l in range(0, L):
        if l == 0:
            model.add(keras.Input(shape=(nn_archi[0],)))
        elif l in range(1, L-1):
            model.add(layers.Dense(nn_archi[l], hl_activation))
        elif l == L-1:
            model.add(layers.Dense(nn_archi[l], ol_activation))
    return model

def compile_model(model, optimizer=keras.optimizers.RMSprop(learning_rate=1e-3), 
                  loss=keras.losses.SparseCategoricalCrossentropy(), 
                  metrics=[keras.metrics.SparseCategoricalAccuracy()]):
    """
    Compiles the model, with specified loss, optimizer and tracking metrics
    Arguments:
        model: the specified nn-model
        optimizer: the optimization function to be used in training the parameters
        loss: the loss function to be computed
        metrics: a list of metrics to keep track of
    Returns: the compiled model
    """
    model.compile(optimizer, loss, metrics)
    
    return model
def predict_new(model, x_new, y_new):
    """
    uses the model to predict accuracy of some new test data
    Arguments: x_new: numpy array of test data, y_new: numpy array of new test data labels
    Returns: array of predicted labels of test data
    """
    x_pred = model.predict(x_new)
    data_no = x_pred.shape[0]    #no of test data
    predictions = []
    for i in range(data_no):
        x_pred_i = np.argmax(x_pred[i])
        if x_pred_i == y_new[i]:
            predictions.append(pred_num(x_pred_i) + "-- correct prediction")
        elif x_pred_i != y_new[i]:
            predictions.append(pred_num(x_pred_i) + "-- wrong prediction")
    return predictions
def pred_num(x_pred_i):
    match x_pred_i:
        case 9: return "Number 9"
        case 8: return "Number 8"
        case 7: return "Number 7"
        case 6: return "Number 6"
        case 5: return "Number 5"
        case 4: return "Number 4"
        case 3: return "Number 3"
        case 2: return "Number 2"
        case 1: return "Number 1"
        case 0: return "Number 0"
def show_img(x_test):
    """
    shows the test data as images
    Arguments: x_test: test data as numpy arrays
    Returns: plots the data as images
    """
    x_img = x_test.reshape(-1, 28, 28)
    fig = plt.figure()
    for i in range(x_img.shape[0]):
        ax = fig.add_subplot(5,5,i+1)
        plt.imshow(x_img[i])
