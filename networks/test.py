import numpy as np 
from PIL import Image
import network
import sys


def main():
    net = network.Network([28*28, 30, 10], '50-10')
    img_path = 'handwritten/2.png'
    print(net.make_dicisions(img_path, reverse=True, show=False))

if __name__ == '__main__':
    main()