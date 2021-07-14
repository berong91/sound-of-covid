import numpy as np
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

import utils
from data_config import SEED, PREFIX_MODEL, POSTFIX_MODEL
from data_config import get_data, get_wav_data, DataGenerator
from data_config import key_col, index_col

for feat in key_col:
    key_col = [feat]
    data = get_data()
    data = get_wav_data(data, feat)
    # Split them out by 0.7 ratio
    (X_test, y_test, X_train, y_train) = utils.prepare_data(data, ratio=0.7, index_col=index_col, key_col=key_col,
                                                            randomize=True, seed=SEED)

    print('X_train.shape:', X_train.shape)
    print('X_test.shape:', X_test.shape)

    print('y_train.shape:', y_train.shape)
    print('y_test.shape:', y_test.shape)

    # determine the input shape
    num_classes = 10
    num_pixels = np.load(X_train[0][0])['arr_0'].shape
    params = {
        'dim': num_pixels,
        'batch_size': 16,
        'n_classes': num_classes,
        'n_channels': 1
    }

    training_generator = DataGenerator(X_train, y_train, **params)
    validation_generator = DataGenerator(X_test, y_test, **params)

    # reshape for input layer
    num_pixels = (num_pixels[0], num_pixels[1], 1)
    ####################################################
    #   Training
    ####################################################
    # define a deep neural network model using the sequential model API
    cnn_model = Sequential(name=feat)
    cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=num_pixels))
    cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(num_classes, activation='softmax'))

    # print a summary of the model
    cnn_model.summary()

    # compile the model using
    # a. Optimizer: gradient descent  with a learning rate of 0.1
    # b. Loss function: categorical_crossentropy
    # c. Metrics: accuracy
    opt = SGD(learning_rate=0.1)
    cnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # fit the model to training data
    # cnn_model.fit(training_generator, use_multiprocessing=True, workers=4)
    # cnn_model.fit(training_generator, epochs=10, verbose=1, workers=2, max_queue_size=2)
    cnn_model.fit(training_generator, epochs=10, verbose=1)

    # evaluate the model on the test data
    loss, acc = cnn_model.evaluate(validation_generator, verbose=1)
    print('Test accuracy = %.4f' % acc)

    # Save model
    model_name = '{prefix}/{index}-{feature}-{postfix}'.format(
        prefix=PREFIX_MODEL,
        index='_'.join(index_col),
        feature=feat,
        postfix=POSTFIX_MODEL
    )
    cnn_model.save(model_name)
