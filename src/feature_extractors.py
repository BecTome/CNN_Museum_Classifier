import src.config as config
import os
import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import umap
from keras.preprocessing.image import ImageDataGenerator


class FeatureExtractor:

    def __init__(self, model_name, checkpoint_path=None):
        self.model_name = model_name
        self.model = None
        self.extractor = None
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
        
        if model_name == "FineTunedVGG16":
            if checkpoint_path is None:
                raise ValueError("Please specify a checkpoint path")
            self.model = self.FineTunedVGG16()
        
        if model_name == "AugmentedScratchModel":
            if checkpoint_path is None:
                raise ValueError("Please specify a checkpoint path")
            self.model = self.AugmentedScratchModel()

        self.build_extractor()
   
    def get_extended_table(self, X):
        _, metadata = self.build_metadata()
        features = self.extractor.predict(X)
        features = features.reshape(features.shape[0], -1)
        df_features = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
        df_features['id'] = X.filenames
        df_data = metadata.merge(df_features, left_on='Image file', right_on='id')
        df_data.drop(columns=['Image file', 'Medium', 'Museum-based instance ID',	'Subset', 'id'], inplace=True)
        return df_data
    
    def fit_preprocess_extended_data(self, X_train, y_train):

        X_train = self.get_extended_table(X_train)
        X_train_prep = X_train.copy()
        scaler = StandardScaler()
        cat_feats = ['Museum']
        num_feats = ['Width', 'Height', 'Product size', 'Aspect ratio']
        ext_feats = [col for col in X_train_prep.columns if 'feature' in col]
        X_train_prep[num_feats + ext_feats] = scaler.fit_transform(X_train_prep[num_feats + ext_feats])

        umap_prep = umap.UMAP(n_components=8)
        new_ext_feats = [f'umap_{i}' for i in range(8)]
        X_train_prep[new_ext_feats] = umap_prep.fit_transform(X_train_prep[ext_feats], y_train)
        X_train_prep.drop(columns=ext_feats, inplace=True)

        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        ohe.fit(X_train_prep[cat_feats])
        ohe_feats = pd.DataFrame(ohe.transform(X_train_prep[cat_feats]), columns=ohe.get_feature_names_out(cat_feats))
        X_train_prep = pd.concat([X_train_prep[num_feats + new_ext_feats], ohe_feats], axis=1)
        
        self.ohe = ohe
        self.scaler = scaler
        self.umap_prep = umap_prep
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        self.ext_feats = ext_feats
        self.new_ext_feats = new_ext_feats
        return X_train_prep
    
    def transform_preprocess_extended_data(self, X_val):
        X_val = self.get_extended_table(X_val)
        X_val_prep = X_val.copy()
        num_feats, cat_feats, ext_feats, new_ext_feats = self.num_feats, self.cat_feats, self.ext_feats, self.new_ext_feats
        X_val_prep[num_feats + ext_feats] = self.scaler.transform(X_val_prep[num_feats + ext_feats])
        X_val_prep[new_ext_feats] = self.umap_prep.transform(X_val_prep[ext_feats])
        X_val_prep.drop(columns=ext_feats, inplace=True)
        ohe_val_feats = pd.DataFrame(self.ohe.transform(X_val_prep[cat_feats]), 
                                     columns=self.ohe.get_feature_names_out(cat_feats))
        X_val_prep = pd.concat([X_val_prep[num_feats + new_ext_feats], pd.DataFrame(ohe_val_feats)], axis=1)
        return X_val_prep

    @staticmethod
    def build_metadata():
        #Readind data directly from the csv file and raw images in order to use data augmentation
        df_labels = pd.read_csv(os.path.join(config.META_PATH, 'MAMe_labels.csv'), header=None, names=['id', 'label'])
        df_labels['label'] = df_labels['label'].str.strip()
        df_info = pd.read_csv(config.META_PATH + 'MAMe_dataset.csv')
        df_info["Medium"] = df_info["Medium"].str.strip()
        df_load_data = df_info.merge(df_labels, right_on='label', left_on='Medium')[['Image file', 'Subset', 'Medium']]
        return df_load_data, df_info

    def build_extractor(self):
        extractor = keras.Model(inputs=self.model.inputs, outputs=self.extraction_layer.output)
        resize_layer = keras.layers.Resizing(self.INPUT_SHAPE[0], self.INPUT_SHAPE[1],
                                              interpolation='bilinear', name='resize')
        preprocess_input_layer = keras.layers.Lambda(self.preprocess_input, name='preprocess_input')
        self.extractor = keras.Sequential([
                                                resize_layer,
                                                preprocess_input_layer,
                                                extractor
                                            ])
        
    def build_dataloaders(self, batch_size=64):
        df_load_data, _ = self.build_metadata()
        datagen = ImageDataGenerator()
        train_generator_df = datagen.flow_from_dataframe(dataframe=df_load_data[df_load_data['Subset'] == 'train'], 
                                                    directory=config.RAW_DATA_PATH,
                                                    x_col="Image file", 
                                                    y_col="Medium", 
                                                    class_mode="sparse", 
                                                    target_size=(config.IMG_SIZE, config.IMG_SIZE), 
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    seed=2020)

        val_generator_df = datagen.flow_from_dataframe(dataframe=df_load_data[df_load_data['Subset'] == 'val'], 
                                                            directory=config.RAW_DATA_PATH,
                                                            x_col="Image file", 
                                                            y_col="Medium", 
                                                            class_mode="sparse", 
                                                            target_size=(config.IMG_SIZE, config.IMG_SIZE), 
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            seed=2020)

        test_generator_df = datagen.flow_from_dataframe(dataframe=df_load_data[df_load_data['Subset'] == 'test'],
                                                    directory=config.RAW_DATA_PATH,
                                                    x_col="Image file",
                                                    y_col="Medium",
                                                    class_mode="sparse",
                                                    target_size=(config.IMG_SIZE, config.IMG_SIZE),
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    seed=2020)
        return train_generator_df, val_generator_df, test_generator_df


    def AugmentedScratchModel(self):
        model = keras.models.load_model(self.checkpoint_path)
        for layer in model.layers:
            layer._name = "_" + layer.name

        self.INPUT_SHAPE = (256, 256, 3)
        self.preprocess_input = lambda x: x / 255.
        self.decode_predictions = None
        return model

    def FineTunedVGG16(self):
        model = keras.models.load_model(self.checkpoint_path)
        for layer in model.layers:
            layer._name = "_" + layer.name

        self.extraction_layer = model.layers[-3]
        self.INPUT_SHAPE = (256, 256, 3)
        self.preprocess_input = lambda x: x / 255.
        self.decode_predictions = None
        return model

    def ResNet50(self):
        from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
        model = ResNet50(weights="imagenet")
        self.INPUT_SHAPE = (224, 224, 3)
        self.preprocess_input = preprocess_input
        self.extraction_layer = model.layers[-2]
        return model

    def VGG16(self):
        from keras.applications.vgg16 import VGG16, preprocess_input
        model = VGG16(weights="imagenet")
        self.INPUT_SHAPE = (224, 224, 3)
        self.preprocess_input = preprocess_input
        self.extraction_layer = model.layers[-2]
        return model

    # def DenseNet121(self):
    #     from keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
    #     model = DenseNet121(weights="imagenet")
    #     self.INPUT_SHAPE = (224, 224, 3)
    #     self.preprocess_input = preprocess_input
    #     self.decode_predictions = decode_predictions
    #     self.extraction_layer = model.layers[-3]
    #     return model

    def EfficientNetB0(self):
        from keras.applications.efficientnet import EfficientNetB0, preprocess_input
        model = EfficientNetB0(weights="imagenet")
        self.INPUT_SHAPE = (224, 224, 3)
        self.preprocess_input = preprocess_input
        self.extraction_layer = model.layers[-2]

        return model

    def EfficientNetB3(self):
        from keras.applications.efficientnet import EfficientNetB3, preprocess_input, decode_predictions
        model = EfficientNetB3(weights="imagenet")
        self.INPUT_SHAPE = (300, 300, 3)
        self.preprocess_input = preprocess_input
        self.extraction_layer = model.layers[-2]
        return model
    

if __name__ == "__main__":
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import src.config as config
    import pandas as pd
    import os

    datagen = ImageDataGenerator()
    BATCH_SIZE = 64

    fe = FeatureExtractor("EfficientNetB0")
    df_load_data, _ = fe.build_metadata()
    train_generator_df = datagen.flow_from_dataframe(dataframe=df_load_data[df_load_data['Subset'] == 'train'], 
                                                        directory=config.RAW_DATA_PATH,
                                                        x_col="Image file", 
                                                        y_col="Medium", 
                                                        class_mode="sparse", 
                                                        target_size=(config.IMG_SIZE, config.IMG_SIZE), 
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False,
                                                        seed=2020)

    val_generator_df = datagen.flow_from_dataframe(dataframe=df_load_data[df_load_data['Subset'] == 'val'], 
                                                        directory=config.RAW_DATA_PATH,
                                                        x_col="Image file", 
                                                        y_col="Medium", 
                                                        class_mode="sparse", 
                                                        target_size=(config.IMG_SIZE, config.IMG_SIZE), 
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False,
                                                        seed=2020)

    y_train, y_val = train_generator_df.classes, val_generator_df.classes

    X_train_prep = fe.fit_preprocess_extended_data(train_generator_df, y_train)
    X_val_prep = fe.transform_preprocess_extended_data(val_generator_df)

    OUTPUT_PATH = 'output/data_trial'
    X_train_prep[:100].to_csv(os.path.join(OUTPUT_PATH, "train.csv"), index=False)
    X_val_prep[:100].to_csv(os.path.join(OUTPUT_PATH, "valid.csv"), index=False)

