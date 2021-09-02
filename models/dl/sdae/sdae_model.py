import warnings
from keras.layers import Dense, Dropout
from keras.layers import Input
from keras.models import Model
import keras.utils.np_utils as npu
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop

from .sdae_model_config import learning_params_template, nb_classes_template

global encoded_layers
def make_layer(layer, x_train, x_test, steps=0, gen=False):
    in_dim = layer['in_dim']
    out_dim = layer['out_dim']
    epochs = layer['epochs']
    batch_size = layer['batch_size']
    optimizer = layer['optimizer']
    enc_act = layer['enc_activation']
    dec_act = layer['dec_activation']

    if optimizer == "sgd":
        optimizer = SGD(lr=layer['lr'],
                        decay=layer['decay'],
                        momentum=layer['momentum'])
    elif optimizer == "adam":
        optimizer = Adam(lr=layer['lr'],
                         decay=layer['decay'])
    elif optimizer == "rmsprop":
        optimizer = RMSprop(lr=layer['lr'],
                            decay=layer['decay'])


    # this is our input placeholder
    input_data = Input(shape=(in_dim,))
    # "encoded" is the encoded representation of the input_data
    encoded = Dense(out_dim, activation=enc_act)(input_data)
    # "decoded" is the lossy reconstruction of the input_data
    decoded = Dense(in_dim, activation=dec_act)(encoded)

    # this model maps an input_data to its reconstruction
    autoencoder = Model(input_data, decoded)

    # this model maps an input_data to its encoded representation
    encoder = Model(input_data, encoded)

    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

    # train layer 1
    if gen:
        (train_steps, test_steps) = steps
        autoencoder.fit_generator(x_train, steps_per_epoch=train_steps, epochs=epochs)
    else:
        autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)

    # encode and decode some digits
    # note that we take them from the *test* set

    if gen:
        (train_steps, test_steps) = steps
        new_x_train1 = encoder.predict_generator(x_train, steps=train_steps)
        new_x_test1 = encoder.predict_generator(x_test, steps=test_steps)
    else:
        new_x_train1 = encoder.predict(x_train)
        new_x_test1 = encoder.predict(x_test)

    weights = encoder.layers[1].get_weights()

    return new_x_train1, new_x_test1, weights

def build_model(learn_params=learning_params_template, nb_classes=nb_classes_template):
    ##注意输入的数据是迭代器
    #(x_train, y_train), (x_test, y_test) = train, test
    layers = learn_params["layers"]

    # Building SAE
    input_data = Input(shape=(layers[0]['in_dim'],))
    prev_layer = input_data

    i = 0
    global encoded_layers
    encoded_layers = []
    for l in layers:
        encoded = Dense(l['out_dim'], activation=l['enc_activation'])(prev_layer)    #多个自编码层之间用了一个全连接层
        i += 1
        encoded_layers.append(i)
        dropout = l["dropout"]
        if dropout > 0.0:
            drop = Dropout(dropout)(encoded)
            i += 1
            prev_layer = drop
        else:
            prev_layer = encoded

    softmax = Dense(nb_classes, activation='softmax')(prev_layer)       #最后一层是个全连接层
    sae = Model(input_data, softmax)
    '''
    if pre_train:
        #这里是在预训练自编码器的encoder-decoder,于是应该提供X的数据
        # Pre-training AEs
        prev_x_train = None
        prev_x_test = None
        for i, l in enumerate(layers):
            if i == 0:
                prev_x_train, prev_x_test, weights = make_layer(l, train_gen, test_gen, steps=steps, gen=True)
            else:
                prev_x_train, prev_x_test, weights = make_layer(l, prev_x_train, prev_x_test)
            sae.layers[encoded_layers[i]].set_weights(weights)
        #print(sae.get_weights())
    '''
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
    sae.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)
    return sae

def pre_train(model,x_train, x_test, learn_params=learning_params_template):
    #这里是在预训练自编码器的encoder-decoder,于是应该提供X的数据
    # Pre-training AEs
    global  encoded_layers
    prev_x_train = None
    prev_x_test = None
    layers = learn_params['layers']
    for i, l in enumerate(layers):
        if i == 0:
            prev_x_train, prev_x_test, weights = make_layer(l, x_train, x_test,gen=False)
        else:
            prev_x_train, prev_x_test, weights = make_layer(l, prev_x_train, prev_x_test)
        model.layers[encoded_layers[i]].set_weights(weights)
    #print(sae.get_weights())

    return  model