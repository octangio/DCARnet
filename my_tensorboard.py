from tensorflow.keras import backend
from tensorflow.keras.callbacks import TensorBoard


class MyTensorBoard(TensorBoard):

    def __init__(self, log_dir,
                 histogram_freq=0,
                 batch_size=128,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 embeddings_data=None,
                 write_leanrning_rate=True):
        super().__init__(log_dir=log_dir,
                         histogram_freq=histogram_freq,
                         batch_size=batch_size,
                         write_graph=write_graph,
                         write_grads=write_grads,
                         write_images=write_images,
                         embeddings_freq=embeddings_freq,
                         embeddings_layer_names=embeddings_layer_names,
                         embeddings_metadata=embeddings_metadata,
                         embeddings_data=embeddings_data)
        self.write_lr = write_leanrning_rate

    def on_epoch_end(self, epoch, logs=None):
        if self.write_lr:
            logs.update({'learning_rate': backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
