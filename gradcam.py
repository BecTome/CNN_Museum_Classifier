## Train a model to classify the images
#
## Import libraries

import os
import numpy as np
import pandas as pd
from tensorflow import keras
np.random.seed(0)

import src.config as config
from src.dataloader import read_val
from src.utils import setup_logger, header
import argparse
from src.analysis import make_gradcam_heatmap, save_and_display_gradcam
import matplotlib.pyplot as plt


## Set up command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--nclass', type=int, default=0)
parser.add_argument('--name', type=str, default='experiment')
parser.add_argument('--outpath', type=str, default=config.OUTPUT_FOLDER)
parser.add_argument('--rescale', type=bool, default=False)

RESCALE = parser.parse_args().rescale
MODEL_PATH  = parser.parse_args().model
N_CLASS = parser.parse_args().nclass
EXPNAME = parser.parse_args().name
OUTPUT_FOLDER = os.path.join(parser.parse_args().outpath, EXPNAME)
                
## Create output folder                 
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

## Set up logger
logger = setup_logger(os.path.join(OUTPUT_FOLDER, 'gradcam_.log'))

## Load data
logger.info(header('LOAD DATA'))

X_val, y_val = read_val()
logger.info(f"VALIDATION SIZE: {X_val.shape}")
logger.info(f"NUMBER OF CLASSES: {np.unique(y_val).shape[0]}")

arr = X_val[y_val == N_CLASS]
N = arr.shape[0]
n_cols = 5

fig, ax = plt.subplots(N // n_cols, n_cols, figsize=(20,(N // 5)*2.5))
ax = ax.ravel()
model_exp = keras.models.load_model(MODEL_PATH)
model_exp.layers[-1].activation = None

conv_layer_name = [layer.name for layer in model_exp.layers if 'conv' in layer.name][-1]

for k in range(N // 10 * 10):
    arr_in = arr[k].copy()
    # Generate class activation heatmap
    if RESCALE:
        heatmap = make_gradcam_heatmap(arr_in / 255., model_exp, conv_layer_name, pred_index=None)
    else:
        heatmap = make_gradcam_heatmap(arr_in, model_exp, conv_layer_name, pred_index=None)

    superimposed = save_and_display_gradcam(arr_in, heatmap)
    ax[k].imshow(superimposed)
    ax[k].axis('off')

fig.suptitle('Heatmaps for class {}'.format(config.LABELS[N_CLASS]))

fig.savefig(os.path.join(OUTPUT_FOLDER, 'gradcam_{}.png'.format(config.LABELS[N_CLASS])))