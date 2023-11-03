## Train a model to classify the images
#
## Import libraries

import os
import numpy as np

from tensorflow import keras
np.random.seed(0)

import src.config as config
from src.dataloader import read_train, read_val
from src.preprocessing import CustomDataGenerator
from src.results import plot_history, plot_cm, class_report,\
                            save_params, write_summary
from src.utils import setup_logger, header
import argparse
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


RAW_DATA_PATH = 'input/data/raw'
META_PATH = 'input/metadata/'

## Set up command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--ep', type=int, default=10)
parser.add_argument('--outpath', type=str, default=config.OUTPUT_FOLDER)

## Set up constants
LEARNING_RATE = parser.parse_args().lr
BATCH_SIZE = parser.parse_args().bs
EPOCHS = parser.parse_args().ep
EVENTNAME = f"lr_{LEARNING_RATE}_bs_{BATCH_SIZE}_ep_{EPOCHS}"
OUTPUT_FOLDER = os.path.join(parser.parse_args().outpath, EVENTNAME)
                
## Create output folder                 
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

## Set up logger
logger = setup_logger(os.path.join(OUTPUT_FOLDER, 'train.log'))

## Write parameters
save_params(OUTPUT_FOLDER, lr=LEARNING_RATE, bs=BATCH_SIZE, ep=EPOCHS)

## Load data
logger.info(header('LOAD DATA'))

#Readind data directly from the csv file and raw images in order to use data augmentation
df_labels = pd.read_csv(os.path.join(META_PATH, 'MAMe_labels.csv'), header=None, names=['id', 'label'])
df_labels['label'] = df_labels['label'].str.strip()
df_info = pd.read_csv(META_PATH + 'MAMe_dataset.csv')
df_info["Medium"] = df_info["Medium"].str.strip()
df_load_data = df_info.merge(df_labels, right_on='label', left_on='Medium')[['Image file', 'Subset', 'id']]

# ImageDataGenerator
datagen = ImageDataGenerator(
                                rotation_range=10, # rotation
                                width_shift_range=0.2, # horizontal shift
                                height_shift_range=0.2, # vertical shift
                                zoom_range=0.2, # zoom
                                horizontal_flip=True, # horizontal flip
                                brightness_range=[0.2,1.2], # brightness
                                rescale=1./255.) 

val_datagen = ImageDataGenerator(rescale=1./255.) # brightness


df_load_data['id'] = df_load_data['id'].astype('str') # requires target in string format

train_generator_df = datagen.flow_from_dataframe(dataframe=df_load_data[df_load_data['Subset'] == 'train'], 
                                              directory=RAW_DATA_PATH,
                                              x_col="Image file", 
                                              y_col="id", 
                                              class_mode="sparse", 
                                              target_size=(256, 256), 
                                              batch_size=BATCH_SIZE,
                                              seed=2020)

val_generator_df = val_datagen.flow_from_dataframe(dataframe=df_load_data[df_load_data['Subset'] == 'val'], 
                                              directory=RAW_DATA_PATH,
                                              x_col="Image file", 
                                              y_col="id", 
                                              class_mode="sparse", 
                                              target_size=(256, 256), 
                                              batch_size=BATCH_SIZE,
                                              seed=2020)




X_train, y_train = read_train()
output_shape = np.unique(y_train).shape[0]
logger.info(f"TRAINING SIZE: {X_train.shape}")
logger.info(f"NUMBER OF CLASSES: {output_shape}")

X_val, y_val = read_val()
logger.info(f"VALIDATION SIZE: {X_val.shape}")
logger.info(f"NUMBER OF CLASSES: {np.unique(y_val).shape[0]}")
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
## Define architecture
logger.info(header('DEFINE MODEL'))

layers = [
            keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, config.N_CHANNELS)),
            keras.layers.Conv2D(32,(3,3), activation = 'relu'),
            keras.layers.MaxPooling2D(pool_size = (4, 4)),
            
            keras.layers.Conv2D(64,(3,3), activation = 'relu'),
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

model.fit_generator(train_generator_df,
                    epochs=EPOCHS,
                    steps_per_epoch=X_train.shape[0]//BATCH_SIZE,  # number of images comprising of one epoch
                    validation_data=val_generator_df,
                    validation_steps=X_val.shape[0]//BATCH_SIZE)

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

logger.info(header('END'))