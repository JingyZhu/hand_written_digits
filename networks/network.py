import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import math
import random
import os, shutil

class Network:

    def __init__(self, layer, directory=None):
        """
        'layer': [list of int],
        'directory': [input, list of (string, string), output] (optional)
        """
        self.input = np.ones([layer[0], 1])
        self.output = np.zeros([layer[-1], 1])
        self.weight = []
        self.bias = []
        if directory is None:
            for i in range(1, len(layer)):
                self.weight.append(np.random.randn(layer[i], layer[i-1]))
                self.bias.append(np.random.randn(layer[i], 1))
        else:
            dirname = os.path.dirname(os.path.realpath(__file__))
            directory = os.path.join(dirname,'config', directory)
            allfile = os.listdir(directory)
            weight = filter(lambda x: x.startswith('weight'), allfile)
            bias = filter(lambda x: x.startswith('bias'), allfile)
            directory = [ (os.path.join(directory, w), os.path.join(directory, b)) for w, b in zip(weight, bias)]
            assert(len(directory) == len(layer)-1)
            for  i in range(0, len(layer)-1):
                self.weight.append(np.load(directory[i][0]))
                self.bias.append(np.load(directory[i][1]))

    def calculate_za(self, input_vec):
        """returns z, a like self.output"""
        z_collection = []
        a_collection = [input_vec]
        for weight, bias in zip(self.weight, self.bias):
            z = np.dot(weight, input_vec) + bias
            z_collection.append(z)
            input_vec = sigmoid(z)
            a_collection.append(input_vec)
        return (z_collection, a_collection)

    def SGD(self, data, mini_batch_size, eta, epoch, test_data = None):
        """data:[list of (img_array, correct_output)]"""
        for time in range(epoch):
            random.shuffle(data)
            mini_batches = [data[k: k+mini_batch_size] for k in range(0, len(data), mini_batch_size)]
            for mini_batch in mini_batches:
                cumulative_w = [np.zeros(weight.shape) for weight in self.weight]
                cumulative_b = [np.zeros(bias.shape) for bias in self.bias]
                for img_vec, y in mini_batch:
                    nabla_w, nabla_b = self.back_prop(img_vec, y)
                    cumulative_w = [w + dw for w, dw in zip(cumulative_w, nabla_w)]
                    cumulative_b = [b + db for b, db in zip(cumulative_b, nabla_b)]
                self.weight = [last - eta*delta/mini_batch_size for last, delta in zip(self.weight, cumulative_w)]
                self.bias = [last - eta*delta/mini_batch_size for last, delta in zip(self.bias, cumulative_b)]
            if test_data:
                self.test(test_data)

    def back_prop(self, img_vec, y):
        nabla_w = [np.zeros(weight.shape) for weight in self.weight]
        nabla_b = [np.zeros(bias.shape) for bias in self.bias]
        z_collection, a_collection = self.calculate_za(img_vec)
        # print(z_collection[-1])
        bp1 = self.cost_function_derivate(a_collection[-1], y) * d_sigmoid(z_collection[-1])
        # print(self.cost_function_derivate(a_collection[-1], y))
        # print(d_sigmoid(z_collection[-1]))
        nabla_b[-1] = bp1
        nabla_w[-1] = np.dot(bp1, a_collection[-2].T)
        for i in range(len(self.weight)-2, -1, -1):  #  [-2] to [0]
            bp1 = np.dot(self.weight[i+1].T, bp1) * d_sigmoid(z_collection[i])
            nabla_b[i] = bp1
            nabla_w[i] = np.dot(bp1, a_collection[i].T)
        return (nabla_w, nabla_b)

    def test(self, test_data):
        correct =  0
        for test in test_data:
            decision = self.output_answer(test[0])
            if decision == np.argmax(test[1]):
                correct += 1
        print('{}/{}'.format(correct, len(test_data)))


    def output_answer(self, input_vec):
        """REQUIRES: input should be a nparray"""
        for i in range(len(self.weight)):
            input_vec = sigmoid(self.weight[i].dot(input_vec) + self.bias[i])
        self.output = input_vec
        decision = np.argmax(self.output)
        return decision
    
    def read_img(self, img_path, size=(28,28), reverse=False, show=False):
        """Convert img to learnable vector"""
        img = Image.open(img_path)
        img = img.convert('L')
        if reverse:
            img = ImageOps.invert(img)
        img = img.resize(size)
        vfilt = np.vectorize(handle_img)
        img = vfilt(np.asarray(img))
        if show:
            img = Image.fromarray(img, 'I')
            img.show()
            return np.resize(np.asarray(img)/255, (self.input.shape[0], 1))
        return np.resize(img/255, (self.input.shape[0], 1))

    def cost_function_derivate(self, a, y):
        """a: Activation, y: correct output"""
        return a-y
    
    def save_to_files(self, path):
        dirname = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dirname, 'config', path)
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)
        for i in range(len(self.weight)):
            np.save(os.path.join(path, 'weight-{}'.format(i)), self.weight[i])
            np.save(os.path.join(path, 'bias-{}'.format(i)), self.bias[i])
    
    def make_dicisions(self, path, reverse=False, show=False):
        array = self.read_img(path, reverse=reverse, show=show)
        return self.output_answer(array)
        

def sigmoid(z):
    """sigmoid function"""
    return 1.0/(1.0+np.exp(-z))

def d_sigmoid(z):
    """sigmoid' function"""
    return sigmoid(z)*(1.0-sigmoid(z))

def handle_img(pixel):
    threashold = 100
    if pixel < threashold:
        return 0
    return pixel