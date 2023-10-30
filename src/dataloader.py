import src.config as config
import os
import numpy as np

def read_train(path=config.TRAIN_PATH):
    train_data = np.load(os.path.join(path, 'train.npz'))
    X_train, y_train = train_data['X'], train_data['y']
    return X_train, y_train

def read_val(path=config.VAL_PATH):
    val_data = np.load(os.path.join(path, 'val.npz'))
    X_val, y_val = val_data['X'], val_data['y']
    return X_val, y_val

def read_test(path=config.TEST_PATH):
    test_data = np.load(os.path.join(path, 'test.npz'))
    X_test, y_test = test_data['X'], test_data['y']
    return X_test, y_test