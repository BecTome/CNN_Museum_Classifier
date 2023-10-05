# CNN_Musseum_Classifier

## Description
Trainig a CNN to classify images from the [Museum Art Medium dataset](https://storage.hpai.bsc.es/mame-dataset/MAMe_data_256.zip) and its [metadata](https://storage.hpai.bsc.es/mame-dataset/MAMe_metadata.zip).

More info about the task in the `4.Autonomous_Lab-CNNs.pdf` file.

## Conda Environment

```
conda env create -f environment.yml
conda env update -f environment.yml
conda env export --no-builds > environment.yml
```

## Folders Structure

```
.
├── input
│   ├── data (not included in the repo)
│   │   ├── raw (raw images)
│   │   ├── test (npz file with X_test and y_test)
│   │   ├── train (npz file with X_train and y_train)
│   │   └── val (npz file with X_val and y_val)
│   ├── metadata
│   └── toy (subsample of the data)
│       ├── test (npz file with X_test and y_test)
│       ├── train (npz file with X_train and y_train)
│       └── val (npz file with X_val and y_val)
├── notebooks
│   ├── complete
│   ├── others
│   └── toy
└── outputs
    ├── models
    ├── others
    └── toy
```

## Solved Issues

1. For the same batch size, for the toy dataset everything ran perfectly but for the complete one there were memory errors. It was solved cleaning the session after each batch pass.

**Solution**:

```
# Generador de datos personalizado basado en Sequence
class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_data = self.data[start:end]
        batch_labels = self.labels[start:end]
        return batch_data, batch_labels

    def on_epoch_end(self):
        # Clear the Keras session to release GPU memory
        K.clear_session()
```