from VGG16 import VGG16
from VGG19 import VGG19
from preprocess import preprocess
from ResNet import ResNet
from ViT import ViT

if __name__ == "__main__":
    preprocess()
    VGG16()
    VGG19()
    ResNet()
    ViT()