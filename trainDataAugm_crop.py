## Train a model to classify the images
#
## Import libraries

import os
import numpy as np

from tensorflow import keras
np.random.seed(0)

import src.config as config
from src.results import plot_history, plot_cm, class_report,\
                            save_params, write_summary
from src.utils import setup_logger, header
from src.preprocessing import center_crop
import argparse
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


RAW_DATA_PATH = 'input/data/raw'
META_PATH = 'input/metadata/'

## Set up command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--name', type=str, default="experiment")
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

#Readind data directly from the csv file and raw images in order to use data augmentation
df_labels = pd.read_csv(os.path.join(META_PATH, 'MAMe_labels.csv'), header=None, names=['id', 'label'])
df_labels['label'] = df_labels['label'].str.strip()
df_info = pd.read_csv(META_PATH + 'MAMe_dataset.csv')
df_info["Medium"] = df_info["Medium"].str.strip()
df_load_data = df_info.merge(df_labels, right_on='label', left_on='Medium')[['Image file', 'Subset', 'Medium']]


datagen = ImageDataGenerator(
                                rotation_range=30,        # Random rotation between -30 and 30 degrees
                                # width_shift_range=0.125,    # Random horizontal shifting (crop)
                                # height_shift_range=0.125,   # Random vertical shifting (crop)
                                horizontal_flip=True,     # Randomly flip images horizontally
                                rescale=1./255.,          # Rescale pixel values to the range [0, 1]
                                preprocessing_function=lambda x: center_crop(x, 200, 200),
                                # featurewise_center=True,  # Normalize by subtracting mean pixel value
                                # featurewise_std_normalization=True,  # Normalize by dividing by standard deviation


)

val_datagen = ImageDataGenerator(rescale=1./255.,
                                preprocessing_function=lambda x: center_crop(x, 200, 200),
                                # featurewise_center=True,  # Normalize by subtracting mean pixel value
                                # featurewise_std_normalization=True,  # Normalize by dividing by standard deviation
)

# val_datagen.mean = 0.5795
# val_datagen.std = 4.22
# df_load_data['id'] = df_load_data['id'].astype('str') # requires target in string format

train_generator_df = datagen.flow_from_dataframe(dataframe=df_load_data[df_load_data['Subset'] == 'train'], 
                                              directory=RAW_DATA_PATH,
                                              x_col="Image file", 
                                              y_col="Medium", 
                                              class_mode="sparse", 
                                              target_size=(200, 200), 
                                              batch_size=BATCH_SIZE,
                                              seed=2020)

val_generator_df = val_datagen.flow_from_dataframe(dataframe=df_load_data[df_load_data['Subset'] == 'val'], 
                                              directory=RAW_DATA_PATH,
                                              x_col="Image file", 
                                              y_col="Medium", 
                                              class_mode="sparse", 
                                              target_size=(200, 200), 
                                              batch_size=BATCH_SIZE,
                                              seed=2020)


test_generator_df = val_datagen.flow_from_dataframe(dataframe=df_load_data[df_load_data['Subset'] == 'val'],
                                                    directory=RAW_DATA_PATH,
                                                    x_col="Image file", 
                                                    y_col="Medium",
                                                    class_mode="sparse",
                                                    target_size=(200, 200), 
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False,
                                                    seed=2020)

y_val = val_generator_df.classes
labels = list(train_generator_df.class_indices.keys())

## Define architecture
logger.info(header('DEFINE MODEL'))

layers = [
        
        keras.Input(shape=(200, 200, config.N_CHANNELS)),
        keras.layers.Conv2D(32,(3,3), activation = 'relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size = (4, 4)),
        keras.layers.Conv2D(64,(3,3), activation = 'relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size = (4, 4)),
        keras.layers.Conv2D(128,(3,3), activation = 'relu'),
        keras.layers.MaxPooling2D(pool_size = (4, 4)),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(29, activation='softmax')
]

model = keras.Sequential(layers)

model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

write_summary(model, OUTPUT_FOLDER)

## Train model
logger.info(header('TRAINING DATA'))

model.fit(train_generator_df,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=val_generator_df)

model_file = os.path.join(OUTPUT_FOLDER, f'model.h5')
model.save(model_file)

## Evaluate model
logger.info(header('WRITE'))
logger.info(f"Write model in: {model_file}")

fig = plot_history(model)
fig.savefig(os.path.join(OUTPUT_FOLDER, f'history_{EVENTNAME}.png'))

y_prob = model.predict(test_generator_df, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

fig = plot_cm(y_val, y_pred, labels=labels)
fig.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrix.jpg'))

fig = plot_cm(y_val, y_pred, labels=labels, pct=True)
fig.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrix_pct.jpg'))

report = class_report(y_val, y_pred, out_path=OUTPUT_FOLDER)

logger.info(header('END'))