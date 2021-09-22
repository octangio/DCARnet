from prepare_data import create_dataset_from_imgs, read_file_list
from my_tensorboard import MyTensorBoard
from my_model import my_load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from DCARnet_model import dcarnet
import os
import argparse
from my_loss import MSE, ssim_loss, ssim, total_loss, contrast, PSNR

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='tensorflow implementation of HARNet')
parser.add_argument('--train_images', type=str, default=0,
                    help='Path of training input')
parser.add_argument('--train_labels', type=str, default=0,
                    help='Path of training label')
parser.add_argument('--valid_images', type=str, default=0,
                    help='Path of validation input')
parser.add_argument('--valid_labels', type=str, default=0,
                    help='Path of validation label')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size')

if __name__ == "__main__":
    epochs = 10000
    args = parser.parse_args()
    train_img_list = read_file_list(args.train_images)
    train_lbl_list = read_file_list(args.train_labels)

    valid_img_list = read_file_list(args.valid_images)
    valid_lbl_list = read_file_list(args.valid_labels)

    # create data generator
    my_training_batch_generator = create_dataset_from_imgs(train_img_list, train_lbl_list, args.batch_size)
    my_validation_batch_generator = create_dataset_from_imgs(valid_img_list, valid_lbl_list, args.batch_size)

    input_img = Input(shape=(76, 76, 1))
    output = dcarnet(input_img)
    model = Model(input_img, output)
    model.compile(optimizer=Adam(lr=0.001), loss=total_loss, metrics=[MSE, ssim_loss, contrast, ssim, PSNR])
    print(model.summary())
    logdir = 'logs'
    tensorboard_visualization = MyTensorBoard(log_dir=logdir,
                                              write_graph=True, write_images=True)
    csv_logger = CSVLogger('{}/training.log'.format(logdir))
    checkpoint = ModelCheckpoint(logdir + '/model_{epoch:02d}_{val_loss:.2f}.hdf5', monitor='val_loss',
                                 verbose=1, save_best_only=True)
    resuce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', epsilon=0.0001,
                                  cooldown=0, min_lr=0.00000001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)

    callbacks = [tensorboard_visualization, checkpoint, resuce_lr, early_stopping, csv_logger]
    [model, init_epoch] = my_load_model(model, logdir=logdir, checkpoint_file='checkpoint.ckp',
                                        custom_objects={})
    # training
    model.fit_generator(generator=my_training_batch_generator,
                        epochs=epochs,
                        initial_epoch=init_epoch,
                        steps_per_epoch=100,
                        validation_steps=30,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=my_validation_batch_generator)
