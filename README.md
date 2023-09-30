# CNN_Musseum_Classifier

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