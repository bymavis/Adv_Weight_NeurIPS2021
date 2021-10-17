import os
import torch
import numpy as np

import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision as tv

from time import time
from torch.autograd import Variable
from attack import FastGradientSignUntargeted
from visualization import VanillaBackprop
#from attack import OdiPgdWhiteBox
from model.resnet_linear_wbn import * 
from model.resnet import *
from utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model, LabelDict

from argument_weight import parser, print_args

layer_list=['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.0.shortcut.0.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.0.shortcut.0.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight', 'layer4.0.shortcut.0.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight']
test_layer=['layer1.1.conv2.weight','layer4.1.conv2.weight']

def load_model(model, model_linear, checkpoints):
    #获取线性模型的参数
    checkpoint= torch.load(checkpoints)
    model.load_state_dict(checkpoint)
    model_dict=model.state_dict()
    model_linear_dict=model_linear.state_dict()
    #print("model list:",model.state_dict().keys())
    model_dict = {k: v for k,v in model_dict.items() if k in model_linear_dict}
    model_linear_dict.update(model_dict)
    model_linear.load_state_dict(model_linear_dict)

    return model_linear



def read_layer_weight(model,args):
    #print(model.state_dict().keys())
    layer_weight_variance=[]
    layer_weight_variance_each=[]
    if not os.path.exists(args.adv+'_layer_weight'):
        os.mkdir(args.adv+'_layer_weight')
    
    #evaluate the orth of conv layer:
    conv_weights = []
    fc_weights = []
    for name, W in model.named_parameters():
        if W.dim() == 4:
            conv_weights.append((name, W))
            
        elif W.dim() == 2:
            fc_weights.append((name, W))
    #print("conv_weights:", conv_weights)
    inter_filter_ortho = {}
    for name, W in conv_weights:
        size = W.size()
        print("size:",size)
        W2d = W.view(size[0], -1)
        W2d = F.normalize(W2d, p=2, dim=1)
        W_WT = torch.mm(W2d, W2d.transpose(0, 1))
        print("W_WT size:",W_WT.size())
        I = torch.eye(W_WT.size()[0], dtype=torch.float32).cuda()
        P = torch.abs(W_WT - I)
        #P = P.sum(dim=1) / size[0]
        #print("p.size():",P.size())
        #inter_filter_ortho[name] = P.cpu().detach().numpy()
        #break
        sigma_n=P.cpu().detach().numpy()
        fig=plt.figure()
        ax = fig.add_subplot(111)
        ax.set_yticks(range(len(sigma_n)))
        #ax.set_yticklabels(classes)
        ax.set_xticks(range(len(sigma_n[0])))
        #ax.set_xticklabels(classes)
        #作图并选择热图的颜色填充风格，这里选择hot
        im = ax.imshow(sigma_n, cmap=plt.cm.Blues_r, vmin=-1, vmax=1)
        #增加右侧的颜色刻度条
        plt.colorbar(im)
        #增加标题
        plt.title("the multi of weight and its transpose__"+str(name))
        save_path=os.path.join(args.adv+'_withoutbn','mul_of_weight_and_transpose_'+str(name)+'.png')
        plt.savefig(save_path,dpi=900)
        plt.close()



class Evaluator():
    def __init__(self, args,logger,attack):
        self.args=args
        self.logger=logger
        self.attack=attack

    def evaluate_std(self,model,model_linear,te_loader):
        self.weight(model,model_linear,te_loader)

    def evaluate_adv(self,model,model_linear,te_loader):
        self.weight(model,model_linear,te_loader)

    def weight(self,model,model_linear,te_loader):
        args=self.args
        logger =self.logger
        
        begin_time=time()

        model.eval()
        model_linear.eval()

        
        batch = 0
        if not os.path.exists(args.adv+'_weight_correlation'):
            os.mkdir(args.adv+'_weight_correlation')

        for data,label in te_loader:
            data, label = tensor2cuda(data), tensor2cuda(label)
            target=model_linear(data)


            adv_data = self.attack.perturb(data, label, 'mean', True)


            x_adv = Variable(adv_data.clone().detach().data,requires_grad=True)
            
            noise = adv_data - data 
            model.eval()
            model_linear.eval()

            output = model_linear(x_adv)
            

            print("the shape of data:", data.size())

            weight = torch.ones((len(data),10, data.size()[1], data.size()[2], data.size()[3]))
            print("the shape of weight:", weight.size())
            for i in range(len(data)):
                for j in range(10):
                    output[i][j].backward(retain_graph=True)
                    weight[i][j] = x_adv.grad[i].detach()
                    x_adv.grad[i].data.zero_()
        



            #evaluate w * w^T
            weight_0 = np.array(weight[0].reshape(10,-1).detach().cpu().numpy())
            norm = np.linalg.norm(x=weight_0,ord=2,axis=1,keepdims=True)
            weight_0_n = weight_0 / norm
            print("the shape of norm:",norm.shape)
            print("the shape of weight_0:",weight_0_n.shape)
            weight_0_n_trans =weight_0_n.transpose()
            print("the shape of weight_0_trans:",weight_0_n_trans.shape)
            sigma_n = np.dot(weight_0_n,weight_0_n_trans)
            print("the shape of sigma",sigma_n.shape)
            #print(sigma_n)

            
            #draw w * w^T
            from matplotlib import cm
            from matplotlib import axes
            fig=plt.figure()
            ax = fig.add_subplot(111)
            ax.set_yticks(range(len(sigma_n)))
            #ax.set_yticklabels(classes)
            ax.set_xticks(range(len(sigma_n[0])))
            #ax.set_xticklabels(classes)
            #作图并选择热图的颜色填充风格，这里选择hot
            im = ax.imshow(sigma_n, cmap=plt.cm.Blues, vmin=-1, vmax=1)
            #增加右侧的颜色刻度条
            plt.colorbar(im)
            #增加标题
            plt.title("Correlation Matrix C")
            save_path=os.path.join(args.adv+'_weight_correlation','mul_of_weight_and_transpose.pdf')
            plt.savefig(save_path,dpi=900)
            plt.close()
            
            print("finish write the multi of weight and its transpose result")






            print('finish a batch work')
            break
            batch += 1







def main(args):


    save_folder = '%s_%s_weight_evaluate' % (args.dataset, args.affix)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

    print_args(args, logger)

    model = ResNet18()
    model_linear = ResNet18_linear_withoutbn()
    if args.adv=='adv':
        model.load_state_dict(torch.load(args.checkpoint_adv))
    elif args.adv=='std':
        model.load_state_dict(torch.load(args.checkpoint_std))
    

    attack = FastGradientSignUntargeted(model, args.epsilon, args.alpha, min_val=0, max_val=1,max_iters=args.k, _type=args.perturbation_type)
    
    if torch.cuda.is_available():
        model.cuda()
        model_linear.cuda()

    evaluator=Evaluator(args,logger,attack)

    if args.todo == 'test':
        te_dataset = tv.datasets.CIFAR10(args.data_root, 
                                       train=False, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        

        if args.adv=='adv':
            model_linear_load = load_model(model, model_linear, args.checkpoint_adv)
            if args.read_layer_weight:
                read_layer_weight(model_linear_load,args)
            if args.evaluate_weight:
                evaluator.evaluate_adv(model,model_linear_load,te_loader)

        elif args.adv=='std':
            model_linear_load = load_model(model, model_linear, args.checkpoint_std)
            if args.read_layer_weight:
                read_layer_weight(model_linear_load,args)
            if args.evaluate_weight:
                evaluator.evaluate_std(model,model_linear_load,te_loader)



    else:
        raise NotImplementedError
    



if __name__ == '__main__':
    args = parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)
