from keras import applications
from src import config
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--outpath', type=str, default=config.OUTPUT_FOLDER)
OUTPUT_FOLDER = os.path.join(parser.parse_args().outpath)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

img_width, img_height = config.IMG_SIZE, config.IMG_SIZE

model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

model_file = os.path.join(OUTPUT_FOLDER, f'VGG16.h5')
model.save(model_file)