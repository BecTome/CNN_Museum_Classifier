from src.dataloader import read_val, read_test

## Specific for ResNet50
from src.imagenet_inference import PretrainedModel


X_val, y_val = read_val()
X_test, y_test = read_test()

# Select one example from each class in X_val and plot them
ls_pretrained_models = ["ResNet50", "VGG16", "DenseNet121", "EfficientNetB0", "EfficientNetB3"]
for model_name in ls_pretrained_models:
    print(f"Plotting predictions for {model_name}")
    pt = PretrainedModel(model_name)

    model = pt.model

    pt.plot_predictions(X_val, y_val, filename=f"{model_name}_val_predictions.png")
    pt.plot_predictions(X_test, y_test, filename=f"{model_name}_test_predictions.png")

ls_checkpoint_paths = [("ScratchModel", "outputs/regularized_model/model.h5"),
                       ("AugmentedScratchModel", "outputs/try_to_overfit_more/lr_1e-05_bs_128_ep_200/model.h5")]

for name, checkpoint_path in ls_checkpoint_paths:
    print(f"Plotting predictions for {name}")

    pt = PretrainedModel(name, checkpoint_path=checkpoint_path)

    model = pt.model

    pt.plot_predictions(X_val, y_val, filename=f"{name}_val_predictions.png")
    pt.plot_predictions(X_test, y_test, filename=f"{name}_test_predictions.png")