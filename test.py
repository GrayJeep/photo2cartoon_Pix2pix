# coding:UTF-8
import os
import argparse
import cv2
import networks
from tensorflow.keras import models
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--photo_path', type=str, help='input photo path')
parser.add_argument('--save_path', type=str, help='cartoon save path')
args = parser.parse_args()

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

if __name__ == '__main__':
    result = None
    generator = networks.Generator()
    generator = models.load_model('model/g_p_500.h5', compile=False)
    img = cv2.cvtColor(cv2.imread(args.photo_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype('float32') / 255

    img = img[None]
    result = generator(img, training=True).numpy()
    result = np.squeeze(result)
    if result is not None:
        result = result * 255
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.save_path, result)
        print('Cartoon portrait has been saved successfully!')
