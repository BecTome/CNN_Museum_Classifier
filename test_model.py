## Train a model to classify the images
#
## Import libraries

import os
import numpy as np

from tensorflow import keras
from tensorflow.keras.regularizers import l1, l2
np.random.seed(0)

import src.config as config
from src.results import plot_history, plot_cm, class_report,\
                            save_params, write_summary
from src.utils import setup_logger, header
from src.preprocessing import center_crop
from src.dataloader import read_test
import argparse
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# set up cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

RAW_DATA_PATH = 'input/data/raw'
META_PATH = 'input/metadata/'

## Set up command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--name', type=str, default='test')
parser.add_argument('--outpath', type=str, default=config.OUTPUT_FOLDER)

## Set up constants

MODEL_PATH = parser.parse_args().model
EXPNAME = parser.parse_args().name
EVENTNAME = f"{MODEL_PATH.replace('/', '_')}"
OUTPUT_FOLDER = os.path.join(parser.parse_args().outpath, EXPNAME, EVENTNAME)

## Create output folder                 
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

## Set up logger
logger = setup_logger(os.path.join(OUTPUT_FOLDER, 'test.log'))

## Load data
logger.info(header('LOAD DATA'))

X_test, y_test = read_test()
logger.info(f"VALIDATION SIZE: {X_test.shape}")
logger.info(f"NUMBER OF CLASSES: {np.unique(y_test).shape[0]}")

model = keras.models.load_model(MODEL_PATH)

## Evaluate model
logger.info(header('WRITE'))
logger.info(f"Write output in: {OUTPUT_FOLDER}")

y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

import matplotlib.pyplot as plt
fig = plot_cm(y_test, y_pred, labels=config.LABELS)
fig.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrix.jpg'))

fig = plot_cm(y_test, y_pred, labels=config.LABELS, pct=True)
fig.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrix_pct.jpg'))

report = class_report(y_test, y_pred, out_path=OUTPUT_FOLDER)

logger.info(header('END'))