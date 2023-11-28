from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from src import config
import os
import pandas as pd
import argparse
from src.utils import setup_logger, header
from src.results import plot_history, plot_history_logloss, plot_cm, class_report,\
                            save_params, write_summary
from src.dataloader import read_train, read_val
import numpy as np


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
EVENTNAME = f"{EXPNAME}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}_ep_{EPOCHS}"
OUTPUT_PATH = os.path.join(parser.parse_args().outpath)
OUTPUT_FOLDER = os.path.join(OUTPUT_PATH, EVENTNAME)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

logger = setup_logger(os.path.join(OUTPUT_FOLDER, 'train.log'))

RAW_DATA_PATH = 'input/data/raw'
META_PATH = 'input/metadata/'
img_width, img_height = config.IMG_SIZE, config.IMG_SIZE
# train_data_dir = "/gpfs/projects/nct00/nct00038/mit67/train"
# validation_data_dir = "/gpfs/projects/nct00/nct00038/mit67/test"
# nb_train_samples = 5359
# nb_validation_samples = 1339 
MODEL_PATH = "pretrained_models/VGG16.h5"
#target_classes = 67

logger.info(header('LOAD DATA'))
#load dataset
df_labels = pd.read_csv(os.path.join(META_PATH, 'MAMe_labels.csv'), header=None, names=['id', 'label'])
df_labels['label'] = df_labels['label'].str.strip()
df_info = pd.read_csv(META_PATH + 'MAMe_dataset.csv')
df_info["Medium"] = df_info["Medium"].str.strip()
df_load_data = df_info.merge(df_labels, right_on='label', left_on='Medium')[['Image file', 'Subset', 'Medium']]

X_train, y_train = read_train()
output_shape = np.unique(y_train).shape[0]
logger.info(f"TRAINING SIZE: {X_train.shape}")
logger.info(f"NUMBER OF CLASSES: {output_shape}")

X_val, y_val = read_val()
logger.info(f"VALIDATION SIZE: {X_val.shape}")
logger.info(f"NUMBER OF CLASSES: {np.unique(y_val).shape[0]}")

# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
        rescale = 1./255)#,
        #horizontal_flip = True,
        #fill_mode = "nearest",
        #zoom_range = 0.3,
        #width_shift_range = 0.3,
        #height_shift_range=0.3,
        #rotation_range=30)

val_datagen = ImageDataGenerator(
        rescale = 1./255)#,
        #horizontal_flip = True,
        #fill_mode = "nearest",
        #zoom_range = 0.3,
        #width_shift_range = 0.3,
        #height_shift_range=0.3,
        #rotation_range=30)

datagen = ImageDataGenerator(
                                # rotation_range=30,        # Random rotation between -30 and 30 degrees
                                # width_shift_range=0.125,    # Random horizontal shifting (crop)
                                # height_shift_range=0.125,   # Random vertical shifting (crop)
                                # horizontal_flip=True,     # Randomly flip images horizontally
                                rescale=1./255.,          # Rescale pixel values to the range [0, 1]
                                # featurewise_center=True,  # Normalize by subtracting mean pixel value
                                # featurewise_std_normalization=True,  # Normalize by dividing by standard deviation

)

val_datagen = ImageDataGenerator(rescale=1./255.,
                                # preprocessing_function=lambda x: center_crop(x, 200, 200),
                                # featurewise_center=True,  # Normalize by subtracting mean pixel value
                                # featurewise_std_normalization=True,  # Normalize by dividing by standard deviation
)
# val_datagen.mean = 0.5
# val_datagen.std = 0.5

# val_datagen.mean = 0.5795
# val_datagen.std = 4.22
# df_load_data['id'] = df_load_data['id'].astype('str') # requires target in string format

train_generator_df = datagen.flow_from_dataframe(dataframe=df_load_data[df_load_data['Subset'] == 'train'], 
                                              directory=RAW_DATA_PATH,
                                              x_col="Image file", 
                                              y_col="Medium", 
                                              class_mode="sparse", 
                                              target_size=(config.IMG_SIZE, config.IMG_SIZE), 
                                              batch_size=BATCH_SIZE,
                                              seed=2020)

val_generator_df = val_datagen.flow_from_dataframe(dataframe=df_load_data[df_load_data['Subset'] == 'val'], 
                                              directory=RAW_DATA_PATH,
                                              x_col="Image file", 
                                              y_col="Medium", 
                                              class_mode="sparse", 
                                              target_size=(config.IMG_SIZE, config.IMG_SIZE), 
                                              batch_size=BATCH_SIZE,
                                              seed=2020)

logger.info(header('LOAD MODEL'))
#model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
model_pre = load_model(MODEL_PATH)

model_pre.summary()
# Freeze the layers which you don't want to train. Here I am freezing the first 10 layers.
for layer in model_pre.layers[:10]:
    layer.trainable = False
logger.info(header('MODIFY MODEL'))

#Adding custom Layers 
x = model_pre.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(config.N_CLASSES, activation="softmax")(x)

# creating the final model 
model_final = Model(model_pre.input, predictions)

# compile the model 
model_final.compile(loss = "sparse_categorical_crossentropy",
                    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE),
                    metrics=["accuracy"])
write_summary(model_final, OUTPUT_FOLDER)

# Save the model according to the conditions  
#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

# Train the model 
history = model_final.fit(train_generator_df,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=val_generator_df)

model_file = os.path.join(OUTPUT_FOLDER, f'model.h5')
model_final.save(model_file)

## Evaluate model
logger.info(header('WRITE'))
logger.info(f"Write model in: {model_file}")

fig = plot_history_logloss(model_final)
fig.savefig(os.path.join(OUTPUT_FOLDER, f'history_{EVENTNAME}.png'))

y_prob = model_final.predict(X_val, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

fig = plot_cm(y_val, y_pred, labels=config.LABELS)
fig.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrix.jpg'))

fig = plot_cm(y_val, y_pred, labels=config.LABELS, pct=True)
fig.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrix_pct.jpg'))

report = class_report(y_val, y_pred, out_path=OUTPUT_FOLDER)

acc_train = model_final.evaluate(X_train, y_train, verbose=0)[1]
acc_val = model_final.evaluate(X_val, y_val, verbose=0)[1]

d_results = {"name": EXPNAME,
             "train_acc": acc_train,
             "val_acc": acc_val,
             "time_per_epoch_s": 21}

df_results = pd.DataFrame(d_results, index=[0])
df_results.to_csv(os.path.join(OUTPUT_FOLDER, 'metrics.csv'), index=False)

def plot_training(history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    #Accuracy plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.title('Training and validation accuracy')
    plt.savefig('fine_tuning_accuracy.pdf')
    plt.close()
    #Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.title('Training and validation loss')
    plt.savefig('fine_tuning_loss.pdf')

plot_training(history)

logger.info(header('END'))


