import warnings
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Input
from keras.models import Sequential
try:
    import hyperas
except ImportError as exp:
    print("Error:{0},\n\t please execute: {1}".format(exp,"pip install hyperas -i https://mirrors.aliyun.com/pypi/simple/"))
    raise exp
from keras.optimizers import SGD, Adam, RMSprop
from .lstm_model_config import learn_params_template,nb_classes_template

def build_model(learn_params=learn_params_template, nb_classes=nb_classes_template):
    input_length = learn_params["maxlen"]
    input_dim = learn_params["nb_features"]
    layers = learn_params["layers"]

    model = Sequential()
    # input_shape = (input_length, input_dim)
    # input_length = maxlen
    # input_dim = nb_features

    if len(layers) == 0:
        raise ("No layers")

    if len(layers) == 1:
        layer = layers[0]
        model.add(LSTM(input_shape=(input_length, input_dim),
                       #batch_input_shape=(batch_size, input_length, input_dim),
                       units=layer['units'],
                       activation=layer['activation'],
                       recurrent_activation=layer['rec_activation'],
                       return_sequences=False,
                       #stateful=True,
                       dropout=layer['dropout']))
        model.add(Dense(units=nb_classes, activation='softmax'))
        return model

    first_l = layers[0]
    last_l = layers[-1]
    middle_ls = layers[1:-1]
    #
    model.add(LSTM(input_shape=(input_length, input_dim),
                   #batch_input_shape=(batch_size, input_length, input_dim),
                   units=first_l['units'],
                   activation=first_l['activation'],
                   recurrent_activation=first_l['rec_activation'],
                   return_sequences=True,
                   #stateful=True,
                   dropout=first_l['dropout']))
    for l in middle_ls:
        model.add(LSTM(units=l['units'],
                       activation=l['activation'],
                       recurrent_activation=l['rec_activation'],
                       return_sequences=True,
                       #stateful=True,
                       dropout=l['dropout']))

    model.add(LSTM(units=last_l['units'],
                   activation=last_l['activation'],
                   recurrent_activation=last_l['rec_activation'],
                   return_sequences=False,
                   #stateful=True,
                   dropout=last_l['dropout']))

    model.add(Dense(units=nb_classes, activation='softmax'))

    if learn_params['optimizer'] == "sgd":
        optimizer = SGD(lr=learn_params['lr'],
                        decay=learn_params['decay'],
                        momentum=0.9,
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


