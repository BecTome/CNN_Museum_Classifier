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

#Readind data directly from the csv file and raw images in order to use data augmentation
df_labels = pd.read_csv(os.path.join(META_PATH, 'MAMe_labels.csv'), header=None, names=['id', 'label'])
df_labels['label'] = df_labels['label'].str.strip()
df_info = pd.read_csv(META_PATH + 'MAMe_dataset.csv')
df_info["Medium"] = df_info["Medium"].str.strip()
df_load_data = df_info.merge(df_labels, right_on='label', left_on='Medium')[['Image file', 'Subset', 'Medium']]

test_datagen = ImageDataGenerator(rescale=1./255.)
                                # preprocessing_function=lambda x: center_crop(x, 200, 200),
                                # featurewise_center=True,  # Normalize by subtracting mean pixel value
                                # featurewise_std_normalization=True,  # Normalize by dividing by standard deviation
# val_datagen.mean = 0.5
# val_datagen.std = 0.5

# val_datagen.mean = 0.5795
# val_datagen.std = 4.22
# df_load_data['id'] = df_load_data['id'].astype('str') # requires target in string format


test_generator_df = test_datagen.flow_from_dataframe(dataframe=df_load_data[df_load_data['Subset'] == 'test'],
                                                    directory=RAW_DATA_PATH,
                                                    x_col="Image file", 
                                                    y_col="Medium",
                                                    class_mode="sparse",
                                                    target_size=(config.IMG_SIZE, config.IMG_SIZE), 
                                                    shuffle=False,
                                                    seed=2020)

y_test = test_generator_df.classes
labels = config.LABELS

model = keras.models.load_model(MODEL_PATH)

## Evaluate model
logger.info(header('WRITE'))
logger.info(f"Write output in: {OUTPUT_FOLDER}")

y_prob = model.predict(test_generator_df, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

import matplotlib.pyplot as plt
fig = plot_cm(y_test, y_pred, labels=labels)
fig.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrix.jpg'))

fig = plot_cm(y_test, y_pred, labels=labels, pct=True)
fig.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrix_pct.jpg'))

report = class_report(y_test, y_pred, out_path=OUTPUT_FOLDER)

logger.info(header('END'))