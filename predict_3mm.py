from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, add, concatenate
import numpy as np
import os
from DCARnet_model import dcarnet
import cv2
from datetime import datetime
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='Tensorflow implementation of DCARnet')
parser.add_argument('--test_data_path', type=str, default=0,
                    help='Path of test input image')
parser.add_argument('--save_path', type=str, default=0,
                    help='The folder path to save output')
parser.add_argument('--logdir', type=str, default=0,
                    help='Path of model weight')
tstart = datetime.now()
print(tstart)
if __name__ == "__main__":
    args = parser.parse_args()
    data_path = args.test_data_path
    save_path = args.save_path
    log_dir = args.logdir
    input_img = Input(shape=(None, None, 1))
    output = dcarnet(input_img)
    model = Model(input_img, output)
    model.load_weights(log_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_list = os.listdir(data_path)
    for i in range(len(file_list)):
        print(i + 1)
        image = data_path + '\\'+file_list[i]
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        name = str(file_list[i])
        out_name = name.split('_o')[0] + '_out.png'
        Y = np.expand_dims(img, 0)
        Y = np.expand_dims(Y, 3) / 255.
        pre = model.predict(Y, batch_size=1) * 255.
        pre[pre[:] > 255] = 255
        pre[pre[:] < 0] = 0
        pre = pre.astype(np.uint8)
        pre = np.squeeze(pre)
        OUTPUT_NAME = save_path + '\\'+out_name
        cv2.imwrite(OUTPUT_NAME, pre)
tend = datetime.now()
print(tend)
time_ = tend - tstart
print(time_)
