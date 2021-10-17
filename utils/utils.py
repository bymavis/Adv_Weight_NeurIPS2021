import os
import json
import logging

import numpy as np

import torch
from torch import nn
ch = torch



class LabelDict():
    def __init__(self, dataset='cifar-10'):
        self.dataset = dataset
        if dataset == 'cifar-10':
            self.label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 
                         4: 'deer',     5: 'dog',        6: 'frog', 7: 'horse',
                         8: 'ship',     9: 'truck'}

        self.class_dict = {v: k for k, v in self.label_dict.items()}

    def label2class(self, label):
        assert label in self.label_dict, 'the label %d is not in %s' % (label, self.dataset)
        return self.label_dict[label]

    def class2label(self, _class):
        assert isinstance(_class, str)
        assert _class in self.class_dict, 'the class %s is not in %s' % (_class, self.dataset)
        return self.class_dict[_class]

def list2cuda(_list):
    array = np.array(_list)
    return numpy2cuda(array)

def numpy2cuda(array):
    tensor = torch.from_numpy(array)

    return tensor2cuda(tensor)

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor

def one_hot(ids, n_class):
    # --------------------- 
    # author：ke1th 
    # source：CSDN 
    # artical：https://blog.csdn.net/u012436149/article/details/77017832 
    b"""
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """

    assert len(ids.shape) == 1, 'the ids should be 1-D'
    # ids = torch.LongTensor(ids).view(-1,1) 

    out_tensor = torch.zeros(len(ids), n_class)

    out_tensor.scatter_(1, ids.cpu().unsqueeze(1), 1.)

    return out_tensor
    
def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()


def create_logger(save_path='', file_type='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_model(model, file_name):
    model.load_state_dict(
            torch.load(file_name, map_location=lambda storage, loc: storage))

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def count_parameters(model):
    # copy from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    # baldassarre.fe's reply
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class FakeReLUM(nn.Module):
    def forward(self, x):
        return FakeReLU.apply(x)

class SequentialWithArgs(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l-1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input