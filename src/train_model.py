import numpy as np
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

import utils
from data_config import get_data, get_wav_data
from data_config import key_col, index_col


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, X_list, y_list, batch_size=32, dim=(32, 32, 32), n_channels=1, n_classes=10):
        'Initialization'
        self.X_list = X_list
        self.y_list = y_list
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_files = [self.X_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_files, indexes)

        return X, y

    def __data_generation(self, list_files, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=float)

        # Generate data
        i = 0
        for f in enumerate(list_files):
            # Store sample
            file = np.load(f[1][0])['arr_0']
            X[i,] = file.reshape(file.shape[0], file.shape[1], 1)
            i += 1

        for i in range(len(indexes)):
            # Store class
            y[i] = self.y_list[indexes[i]]

        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_list))


data = get_data()
data = get_wav_data(data)
# Split them out by 0.7 ratio
(X_test, y_test, X_train, y_train) = utils.prepare_data(data, ratio=0.7, index_col=index_col, key_col=key_col)

print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)

print('y_train.shape:', y_train.shape)
print('y_test.shape:', y_test.shape)

# determine the input shape
num_classes = 10
num_pixels = np.load(X_train[0][0])['arr_0'].shape
params = {
    'dim': num_pixels,
    'batch_size': 32,
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
# Layer 0: input layer specifying the dimension of each sample
# Layer 1: 2D convolution layer with 32 filters, each filter of dimension 3x3, using ReLU activation function
# Layer 2: 2D max pooling layer with filter dimension of 2 x 2
# Layer 3: Flatten the images into a column vector
# Layer 4: Fully connected NN layer with n = 100 nodes, g = ReLU
# Layer 5: Fully connected NN layer with n = num_classes nodes, g = softmax
cnn_model = Sequential()
cnn_model.add(Input(shape=num_pixels))
cnn_model.add(Conv2D(32, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(100, activation='relu'))
cnn_model.add(Dense(num_classes, activation='softmax'))

# print a summary of the model
cnn_model.summary()

# compile the model using
# a. Optimizer: gradient descent  with a learning rate of 0.1
# b. Loss function: sparse_categorical_crossentropy
# c. Metrics: accuracy
opt = SGD(learning_rate=0.1)
cnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model to training data
# cnn_model.fit(X_train, y_train, epochs=10, verbose=1)
# cnn_model.fit(training_generator, use_multiprocessing=True, workers=4)
cnn_model.fit(training_generator, epochs=10, verbose=1)

# evaluate the model on the test data
loss, acc = cnn_model.evaluate(validation_generator, verbose=1)
print('Test accuracy = %.4f' % acc)

cnn_model.save('./train_model')
