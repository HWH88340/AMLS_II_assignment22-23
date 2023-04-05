import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
import cv2
from VGG16 import VGG16
from VGG19 import VGG19
from preprocess import preprocess
from ResNet import ResNet
# from ViT import ViT

if __name__ == "__main__":
    # preprocess()
    VGG16()
    # VGG19()
    # ResNet()
    # ViT()