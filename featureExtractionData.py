## Specific for ResNet50
from src.feature_extractors import FeatureExtractor
import src.config as config
import pandas as pd
import os

BATCH_SIZE = 64

# deactivate gpu
# Select one example from each class in X_val and plot them
ls_pretrained_models = ["ResNet50", "VGG16", "EfficientNetB0", "EfficientNetB3"]
for model_name in ls_pretrained_models:
    print(f"Generating Features for {model_name}")
    fe = FeatureExtractor("EfficientNetB0")
    train_generator_df, val_generator_df, test_generator_df = fe.build_dataloaders(batch_size=BATCH_SIZE)

    y_train = train_generator_df.classes
    y_val = val_generator_df.classes
    y_test = test_generator_df.classes


    X_train_prep = fe.fit_preprocess_extended_data(train_generator_df, y_train)
    X_val_prep = fe.transform_preprocess_extended_data(val_generator_df)
    X_test_prep = fe.transform_preprocess_extended_data(test_generator_df)
    
    path = os.path.join(config.OUTPUT_FEATTOP_DATA, model_name)
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


# ls_checkpoint_paths = [("ScratchModel", "outputs/regularized_model/model.h5"),
#                        ("AugmentedScratchModel", "outputs/try_to_overfit_more/lr_1e-05_bs_128_ep_200/model.h5")]

# for name, checkpoint_path in ls_checkpoint_paths:
#     print(f"Plotting predictions for {name}")

#     pt = PretrainedModel(name, checkpoint_path=checkpoint_path)

#     model = pt.model

#     pt.plot_predictions(X_val, y_val, filename=f"{name}_val_predictions.png")
#     pt.plot_predictions(X_test, y_test, filename=f"{name}_test_predictions.png")