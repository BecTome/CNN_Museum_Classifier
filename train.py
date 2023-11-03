## Train a model to classify the images
#
## Import libraries

import os
import numpy as np
import pandas as pd
import time

from tensorflow import keras
np.random.seed(0)

import src.config as config
from src.dataloader import read_train, read_val
from src.preprocessing import CustomDataGenerator
from src.results import plot_history, plot_cm, class_report,\
                            save_params, write_summary
from src.utils import setup_logger, header
import argparse

## Set up command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='experiment')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--ep', type=int, default=10)
parser.add_argument('--outpath', type=str, default=config.OUTPUT_FOLDER)

## Set up constants
LEARNING_RATE = parser.parse_args().lr
BATCH_SIZE = parser.parse_args().bs
EPOCHS = parser.parse_args().ep
EXPNAME = parser.parse_args().name
EVENTNAME = f"lr_{LEARNING_RATE}_bs_{BATCH_SIZE}_ep_{EPOCHS}"
OUTPUT_FOLDER = os.path.join(parser.parse_args().outpath, EXPNAME, EVENTNAME)
                
## Create output folder                 
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

## Set up logger
logger = setup_logger(os.path.join(OUTPUT_FOLDER, 'train.log'))

## Write parameters
save_params(OUTPUT_FOLDER, lr=LEARNING_RATE, bs=BATCH_SIZE, ep=EPOCHS)

## Load data
logger.info(header('LOAD DATA'))

X_train, y_train = read_train()
output_shape = np.unique(y_train).shape[0]
logger.info(f"TRAINING SIZE: {X_train.shape}")
logger.info(f"NUMBER OF CLASSES: {output_shape}")

X_val, y_val = read_val()
logger.info(f"VALIDATION SIZE: {X_val.shape}")
logger.info(f"NUMBER OF CLASSES: {np.unique(y_val).shape[0]}")

## Define architecture
logger.info(header('DEFINE MODEL'))

layers = [
            keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, config.N_CHANNELS)),
            keras.layers.Rescaling(1./255.),
            keras.layers.Conv2D(1,(3,3), activation = 'relu'),
            keras.layers.MaxPooling2D(pool_size = (4, 4)),

            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(output_shape, activation='softmax')
]

model = keras.Sequential(layers)

model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

write_summary(model, OUTPUT_FOLDER)

## Train model
logger.info(header('TRAINING DATA'))
train_generator = CustomDataGenerator(X_train, y_train, batch_size=BATCH_SIZE)


model.fit(train_generator, epochs=EPOCHS, validation_data=(X_val, y_val))
model_file = os.path.join(OUTPUT_FOLDER, f'model.keras')
model.save(model_file)

## Evaluate model
logger.info(header('WRITE'))
logger.info(f"Write model in: {model_file}")

fig = plot_history(model)
fig.savefig(os.path.join(OUTPUT_FOLDER, f'history_{EVENTNAME}.png'))

y_prob = model.predict(X_val, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

fig = plot_cm(y_val, y_pred, labels=config.LABELS)
fig.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrix.jpg'))

fig = plot_cm(y_val, y_pred, labels=config.LABELS, pct=True)
fig.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrix_pct.jpg'))

report = class_report(y_val, y_pred, out_path=OUTPUT_FOLDER)

acc_train = model.evaluate(X_train, y_train, verbose=0)[1]
acc_val = model.evaluate(X_val, y_val, verbose=0)[1]

d_results = {"name": EXPNAME,
             "train_acc": acc_train,
             "val_acc": acc_val,
             "time_per_epoch_s": 21}

df_results = pd.DataFrame(d_results, index=[0])
df_results.to_csv(os.path.join(OUTPUT_FOLDER, 'metrics.csv'), index=False)

logger.info(header('END'))