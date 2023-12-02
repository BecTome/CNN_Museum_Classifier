## Specific for ResNet50
from src.feature_extractors import FeatureExtractor
import src.config as config
import pandas as pd
import os

BATCH_SIZE = 16

# deactivate gpu
# Select one example from each class in X_val and plot them
# ls_pretrained_models = ["ResNet50", "VGG16", "EfficientNetB0", "EfficientNetB3"]
# for model_name in ls_pretrained_models:
#     print(f"Generating Features for {model_name}")
#     fe = FeatureExtractor("EfficientNetB0")
#     train_generator_df, val_generator_df, test_generator_df = fe.build_dataloaders(batch_size=BATCH_SIZE)

#     y_train = train_generator_df.classes
#     y_val = val_generator_df.classes
#     y_test = test_generator_df.classes


#     X_train_prep = fe.fit_preprocess_extended_data(train_generator_df, y_train)
#     X_val_prep = fe.transform_preprocess_extended_data(val_generator_df)
#     X_test_prep = fe.transform_preprocess_extended_data(test_generator_df)
    
#     path = os.path.join(config.OUTPUT_FEATTOP_DATA, model_name)
#     if not os.path.exists(path):
#         os.makedirs(path)

#     X_train_prep.to_csv(os.path.join(path, "train.csv"), index=False)
#     X_val_prep.to_csv(os.path.join(path, "valid.csv"), index=False)
#     X_test_prep.to_csv(os.path.join(path, "test.csv"), index=False)

#     df_y_train = pd.DataFrame(y_train, columns=["label"])
#     df_y_val = pd.DataFrame(y_val, columns=["label"])
#     df_y_test = pd.DataFrame(y_test, columns=["label"])

#     df_y_train.to_csv(os.path.join(path, "y_train.csv"), index=False)
#     df_y_val.to_csv(os.path.join(path, "y_valid.csv"), index=False)
#     df_y_test.to_csv(os.path.join(path, "y_test.csv"), index=False)


ls_checkpoint_paths = [("FineTunedVGG16", "outputs/best_lr_4.5e-06_bs_300_ep_60_l2_0.15/model.h5")]
for name, checkpoint_path in ls_checkpoint_paths:
    print(f"Generating Features for {name}")

    fe = FeatureExtractor(name, checkpoint_path=checkpoint_path)

    train_generator_df, val_generator_df, test_generator_df = fe.build_dataloaders(batch_size=BATCH_SIZE)

    y_train = train_generator_df.classes
    y_val = val_generator_df.classes
    y_test = test_generator_df.classes


    X_train_prep = fe.fit_preprocess_extended_data(train_generator_df, y_train)
    X_val_prep = fe.transform_preprocess_extended_data(val_generator_df)
    X_test_prep = fe.transform_preprocess_extended_data(test_generator_df)
    
    path = os.path.join(config.OUTPUT_FEATEXT_DATA, name)
    if not os.path.exists(path):
        os.makedirs(path)

    X_train_prep.to_csv(os.path.join(path, "train.csv"), index=False)
    X_val_prep.to_csv(os.path.join(path, "valid.csv"), index=False)
    X_test_prep.to_csv(os.path.join(path, "test.csv"), index=False)

    df_y_train = pd.DataFrame(y_train, columns=["label"])
    df_y_val = pd.DataFrame(y_val, columns=["label"])
    df_y_test = pd.DataFrame(y_test, columns=["label"])

    df_y_train.to_csv(os.path.join(path, "y_train.csv"), index=False)
    df_y_val.to_csv(os.path.join(path, "y_valid.csv"), index=False)
    df_y_test.to_csv(os.path.join(path, "y_test.csv"), index=False)