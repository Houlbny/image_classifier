import sys
import tensorflow as tf
import os

filepath = sys.argv[1]

f = os.listdir(filepath)

for i in f:
    image_path = filepath + i
    print(image_path)