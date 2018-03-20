import os
import numpy as np 
from PIL import Image
import network
import glob

def read_pairs(path, num, size=(28,28)):
    rst = []
    img_paths = os.listdir(path)
    img_paths = filter(lambda x: x.split('.')[-1]=='png', img_paths)
    img_paths = sorted(img_paths, key=lambda x: int(x.split('.')[0]))
    img_paths = img_paths[:num]
    label_paths = glob.glob(os.path.join(path, '*.txt'))
    labels = open(label_paths[0], 'r').read().split(',')
    labels = labels[:num]
    labels = list(map(int, labels))
    for img_path, label in zip(img_paths, labels):
        img = Image.open(os.path.join(path, img_path))
        img = img.resize(size)
        img = img.convert('L')
        y_vec = np.zeros((10, 1))
        y_vec[label][0] = 1
        rst.append((np.reshape(np.asarray(img)/255, (size[0]*size[1], 1)), y_vec))
    return rst

def main():
    first = network.Network([28*28, 50, 10], '50-10')
    data = read_pairs('trainning', 60000)
    testing = read_pairs('test', 10000)
    first.SGD(data, 100, 3.0, 30, testing)
    first.save_to_files('50-10')

if __name__ == '__main__':
    main()