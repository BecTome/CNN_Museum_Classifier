import skimage
import numpy as np
import matplotlib.pyplot as plt
import src.config as config
import os
from src.preprocessing import CustomDataGenerator
from src.analysis import make_gradcam_heatmap, save_and_display_gradcam
import keras

class PretrainedModel:

    def __init__(self, model_name, checkpoint_path=None):
        self.model_name = model_name
        self.model = None
        self.INPUT_SHAPE = None
        self.preprocess_input = None
        self.decode_predictions = None
        self.checkpoint_path = checkpoint_path

        if model_name == "ResNet50":
            self.model = self.ResNet50()
        
        if model_name == "VGG16":
            self.model = self.VGG16()

        if model_name == "DenseNet121":
            self.model = self.DenseNet121()
        
        if model_name == "EfficientNetB0":
            self.model = self.EfficientNetB0()

        if model_name == "EfficientNetB3":
            self.model = self.EfficientNetB3()
        
        if model_name == "ScratchModel":
            if checkpoint_path is None:
                raise ValueError("Please specify a checkpoint path")
            self.model = self.ScratchModel()
        
        if model_name == "AugmentedScratchModel":
            if checkpoint_path is None:
                raise ValueError("Please specify a checkpoint path")
            self.model = self.AugmentedScratchModel()
   
    def preprocess_input_custom(self, X):
            
        X_res = [self.preprocess_input(skimage.transform.resize(img, self.INPUT_SHAPE, preserve_range=True))
                 for img in X]
        return np.array(X_res)
    
    def predict_image(self, img):  
        img = self.preprocess_input_custom([img])[0]
        img_pred = self.model.predict(np.expand_dims(img, axis=0), verbose=0)

        if (self.decode_predictions is not None):
            prediction = self.decode_predictions(img_pred, top=1)[0][0]
            img_pred_lab = prediction[1]
            img_pred_prob = prediction[2]
        else:
            idx_label = np.argmax(img_pred, axis=1)[0]
            img_pred_lab = config.LABELS[idx_label]
            img_pred_prob = np.max(img_pred, axis=1)[0]
        return img_pred_lab, img_pred_prob
    
    def plot_predictions(self, X_val, y_val, 
                         n_classes=config.N_CLASSES,
                         output_path=config.OUTPUT_FEATEXT_PATH,
                         filename="predictions.png"):
        
        fig, ax = plt.subplots(6, 5, figsize=(10, 15))
        ax = ax.ravel()
        for i in range(n_classes):
            img = X_val[y_val == i][0]
            gradcam = self.gradcam(img)
            ax[i].imshow(gradcam)
            label_pred = self.predict_image(img)
            ax[i].set_title(f"{config.LABELS[i]}\n{label_pred[0]} ({label_pred[1]:.2f})")
            ax[i].set_axis_off()

        fig.suptitle(f"Predictions from {self.model_name}", fontsize=16)
        fig.tight_layout()

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        fig.savefig(os.path.join(output_path, filename))
        return fig, ax
    
    def predict(self, X):
        X = self.preprocess_input_custom(X)
        gen = CustomDataGenerator(X, np.zeros(len(X)), batch_size=32)
        y_pred = np.argmax(self.model.predict(gen), axis=1)
        return y_pred
    
    def evaluate(self, X, y):
        X = self.preprocess_input_custom(X)
        # model = self.adapt_last_layer(self.model)
        gen = CustomDataGenerator(X, y, batch_size=32)
        y_pred = np.argmax(self.model.predict(gen), axis=1)
        from sklearn.metrics import adjusted_rand_score
        return adjusted_rand_score(y, y_pred)

    def gradcam(self, img):
        import re
        img_orig = img.copy()
        img = self.preprocess_input_custom([img])[0]
        conv_layer_name = [layer.name for layer in self.model.layers if re.match(".*_[Cc]onv.*$", layer.name)][-1]
        heatmap = make_gradcam_heatmap(img, self.model, conv_layer_name, pred_index=None)
        superimposed = save_and_display_gradcam(img_orig, heatmap)
        return superimposed
    

    def AugmentedScratchModel(self):
        model = keras.models.load_model(self.checkpoint_path)
        for layer in model.layers:
            layer._name = "_" + layer.name

        self.INPUT_SHAPE = (256, 256, 3)
        self.preprocess_input = lambda x: x / 255.
        self.decode_predictions = None
        return model

    def ScratchModel(self):
        model = keras.models.load_model(self.checkpoint_path)
        for layer in model.layers:
            layer._name = "_" + layer.name

        self.INPUT_SHAPE = (256, 256, 3)
        self.preprocess_input = lambda x: x
        self.decode_predictions = None
        return model

    def ResNet50(self):
        from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
        model = ResNet50(weights="imagenet")
        self.INPUT_SHAPE = (224, 224, 3)
        self.preprocess_input = preprocess_input
        self.decode_predictions = decode_predictions
        return model

    def VGG16(self):
        from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
        model = VGG16(weights="imagenet")
        self.INPUT_SHAPE = (224, 224, 3)
        self.preprocess_input = preprocess_input
        self.decode_predictions = decode_predictions
        return model

    def DenseNet121(self):
        from keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
        model = DenseNet121(weights="imagenet")
        self.INPUT_SHAPE = (224, 224, 3)
        self.preprocess_input = preprocess_input
        self.decode_predictions = decode_predictions
        return model

    def EfficientNetB0(self):
        from keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
        model = EfficientNetB0(weights="imagenet")
        self.INPUT_SHAPE = (224, 224, 3)
        self.preprocess_input = preprocess_input
        self.decode_predictions = decode_predictions
        return model

    def EfficientNetB3(self):
        from keras.applications.efficientnet import EfficientNetB3, preprocess_input, decode_predictions
        model = EfficientNetB3(weights="imagenet")
        self.INPUT_SHAPE = (300, 300, 3)
        self.preprocess_input = preprocess_input
        self.decode_predictions = decode_predictions
        return model