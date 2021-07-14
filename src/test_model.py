import numpy as np
from tensorflow.python.keras.saving.save import load_model

import utils
from data_config import PREFIX_MODEL, POSTFIX_MODEL, SEED, DataGenerator
from data_config import get_data, get_wav_data
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
    num_pixels = np.load(X_test[0][0])['arr_0'].shape
    params = {
        'dim': num_pixels,
        'batch_size': 32,
        'n_classes': num_classes,
        'n_channels': 1
    }

    validation_generator = DataGenerator(X_test, y_test, **params)

    # Load model
    model_name = '{prefix}/{index}-{feature}-{postfix}'.format(
        prefix=PREFIX_MODEL,
        index='_'.join(index_col),
        feature=feat,
        postfix=POSTFIX_MODEL
    )
    model = load_model(model_name, custom_objects=None, compile=True, options=None)
    model.summary()

    loss, acc = model.evaluate(validation_generator, verbose=1)
    print('Test accuracy = %.4f' % acc)
    print('###########################################')
