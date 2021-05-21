import warnings
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam, SGD
from keras import regularizers
from keras.layers.recurrent import LSTM
from .cnn_model_config import  learning_params_template,nb_classes_template
from keras import optimizers

def build_model(learn_params=learning_params_template, nb_classes=nb_classes_template):
    input_length = learn_params["maxlen"]
    input_dim = learn_params["nb_features"]
    layers = learn_params["layers"]

    model = Sequential()

    maxlen = input_length
    max_features = input_dim

    if len(layers) == 0:
        raise("No layers")

    first_l = layers[0]
    rest_l = layers[1:]

    # First layer
    if first_l["name"] == 'dropout':
        model.add(Dropout(input_shape=(maxlen, max_features), rate=first_l['rate']))
    elif first_l["name"] == 'conv':
        model.add(Conv1D(filters=first_l['filters'],
                         kernel_size=first_l['kernel_size'],
                         padding='valid',
                         activation=first_l['activation'],
                         strides=first_l['stride']))

    # Middle layers (conv, dropout, pooling, dense, lstm.....)
    for l in rest_l:
        if l["name"] == 'maxpooling':
            model.add(MaxPooling1D(pool_size=l['pool_size'], padding='valid'))
        elif l["name"] == 'conv':
            model.add(Conv1D(filters=l['filters'],
                             kernel_size=l['kernel_size'],
                             padding='valid',
                             activation=l['activation'],
                             strides=l['stride']))
        elif l["name"] == 'dropout':
            model.add(Dropout(rate=l['rate']))
        elif l["name"] == 'lstm':
            model.add(LSTM(l['units']))
        elif l["name"] == 'flatten':
            model.add(Flatten())
        elif l["name"] == 'dense':
            if l['regularization'] > 0.0:
                model.add(Dense(units=l['units'], activation=l['activation'],
                            kernel_regularizer=regularizers.l2(last_l['regularization']),
                            activity_regularizer=regularizers.l1(last_l['regularization'])))
            else:
                model.add(Dense(units=l['units'], activation=l['activation']))


    learn_params = learning_params_template
    if learn_params['optimizer'] == "sgd":
        optimizer = SGD(lr=learn_params['lr'],
                        decay=learn_params['decay'],
                        momentum=learn_params['momentum'],
                        nesterov=True)
    elif learn_params['optimizer'] == "adam":
        optimizer = Adam(lr=learn_params['lr'],
                         decay=learn_params['decay'])
    else:  # elif learn_params['optimizer'] == "rmsprop":
        optimizer = RMSprop(lr=learn_params['lr'],
                            decay=learn_params['decay'])
    metrics=['accuracy']
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)
    return model

