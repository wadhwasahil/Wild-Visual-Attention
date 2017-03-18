import read_data
import tensorflow as tf
import numpy as np


data = read_data.read()

img = next(data)
img = np.resize(img, (32, 32, 3))
print(img.shape)