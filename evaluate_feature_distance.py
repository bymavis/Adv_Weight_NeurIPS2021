import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torchvision as tv

from time import time
import copy
from attack import FastGradientSignUntargeted
from torch.autograd import Variable
from model.resnet import *
from utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model
import argparse

parser = argparse.ArgumentParser(description='Video Summarization')
parser.add_argument('--todo', choices=['train', 'valid', 'test', 'visualize'], default='test',
                    help='what behavior want to do: train | valid | test | visualize')
parser.add_argument('--dataset', default='cifar-10', help='use what dataset')
parser.add_argument('--data_root', default='/data', 
                    help='the directory to save the dataset')
parser.add_argument('--log_root', default='log', 
                    help='the directory to save the logs or other imformations (e.g. images)')
parser.add_argument('--load_std_checkpoint', default='checkpoint/cifar-10_std_train_baseline/last_model.pth')
parser.add_argument('--load_adv_checkpoint', default='checkpoint/cifar-10_adv_train_baseline/last_model.pth')
parser.add_argument('--adv', choices=['adv','std','trades'],default='std',help='what type of model to do: adv | std | trades')
parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
parser.add_argument('--affix', default='pgd20', help='the affix for the save folder')

# parameters for generating adversarial examples
parser.add_argument('--epsilon', '-e', type=float, default=8/255, 
                    help='maximum perturbation of adversaries (8/255=0.0314)')
parser.add_argument('--alpha', '-a', type=float, default=0.007, 
                    help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
parser.add_argument('--beta', type=float, default=1, 
                    help='the weight of regularization weight cluster loss')
parser.add_argument('--k', '-k', type=int, default=10, 
                    help='maximum iteration when generating adversarial examples')
parser.add_argument('--attack_method', choices=['PGD', 'FGSM'], default='PGD')

parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
parser.add_argument('--max_epoch', '-m_e', type=int, default=200, help='the maximum numbers of the model see a sample')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-1, help='learning rate')
parser.add_argument('--weight_decay', '-w', type=float, default=2e-4, 
                    help='the parameter of l2 restriction for weights')


parser.add_argument('--gpu', '-g', default='3', help='which gpu to use')
parser.add_argument('--n_eval_step', type=int, default=100, 
                    help='number of iteration per one evaluation')
parser.add_argument('--n_checkpoint_step', type=int, default=4000, 
                    help='number of iteration to save a checkpoint')
parser.add_argument('--n_store_image_step', type=int, default=4000, 
                    help='number of iteration to save adversaries')
parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf', 
                    help='the type of the perturbation (linf or l2)')

parser.add_argument('--adv_train', action='store_true')
parser.add_argument('--resume_to_checkpoint', action='store_true')


def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))

args = parser.parse_args()

class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def standard_train(self, model, model_linear, tr_loader, va_loader=None):
        self.train(model, model_linear, tr_loader, va_loader, False)

    def adversarial_train(self, model, model_linear, tr_loader, va_loader=None):
        self.train(model, model_linear, tr_loader, va_loader, True)

    def train(self, model, model_linear, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger

        opt = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, 
                                                         milestones=[75, 90], 
                                                         gamma=0.1)
        _iter = 0
       
        #begin training!

        begin_time = time()
        best_pred = 0.0
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        if args.resume_to_checkpoint:
            model.load_state_dict(torch.load(args.checkpoint_std))


        for epoch in range(1, args.max_epoch+1):

            model.train()
            scheduler.step()
            #reg_done=False
            for data, label in tr_loader:

                data, label = tensor2cuda(data), tensor2cuda(label)                       
                
                if adv_train:

                    adv_data = self.attack.perturb(data, label, 'mean', True)
                    model.train()
                    output=model(adv_data)

                else:
    
                    
                    model.train()
                    output = model(data)
                    
                class1_avg = ((output[:,0]+output[:,1]+output[:,8]+output[:,9])/4).detach()
                class2_avg = ((output[:,2]+output[:,3]+output[:,4]+output[:,5]+output[:,6]+output[:,7])/6).detach()
                #reg_loss_3= loss_fn(class1_avg.float(),class2_avg.float())
                reg_loss_1 = (loss_fn(output[:,0],class1_avg)+loss_fn(output[:,1],class1_avg)+loss_fn(output[:,8],class1_avg)+loss_fn(output[:,9],class1_avg))/4
                reg_loss_2 = (loss_fn(output[:,2],class2_avg)+loss_fn(output[:,3],class2_avg)+loss_fn(output[:,4],class2_avg)+loss_fn(output[:,5],class2_avg)+loss_fn(output[:,6],class2_avg)+loss_fn(output[:,7],class2_avg))/6
                reg_loss=0.5*reg_loss_1+0.5*reg_loss_2
                print("reg_loss_1:",reg_loss_1)
                print("reg_loss_2:",reg_loss_2)
        
                adv_loss = F.cross_entropy(output, label)
                
                print("reg_loss:",reg_loss)
                print("adv_loss:",adv_loss)
                loss=args.beta*reg_loss+adv_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

                
                if _iter % args.n_checkpoint_step == 0:
                    file_name = os.path.join(args.model_folder, 'checkpoint_%d.pth' % _iter)
                    save_model(model, file_name)

                _iter += 1

            if va_loader is not None:
                t1 = time()
                va_acc_20, va_adv_acc_20 = self.test_white_box(model, va_loader)
                va_acc_20, va_adv_acc_20 = va_acc_20 * 100.0, va_adv_acc_20 * 100.0
                t2 = time()
                new_pred = va_adv_acc_20
                if new_pred > best_pred:
                    best_pred = new_pred
                    file_name = os.path.join(args.model_folder, 'best_model_20.pth')
                    save_model(model, file_name)
                if epoch == args.max_epoch:
                    file_name = os.path.join(args.model_folder, 'last_model.pth')
                    save_model(model, file_name)

                logger.info('\n'+'='*20 +' evaluation at epoch: %d iteration: %d '%(epoch, _iter) \
                    +'='*20)
                logger.info('test acc 20: %.3f %%, test adv acc 20: %.3f %%,  spent: %.3f' % (
                    va_acc_20, va_adv_acc_20,  t2-t1))
                logger.info('='*28+' end of evaluation '+'='*28+'\n')


    


    def get_reg_weight(self,model_linear,adv_data):
        '''
        :param model:
        :return:
        '''
        model_linear.eval()
        output = model_linear(adv_data)
        #calculate the weight
        weight = torch.ones((len(adv_data),10, adv_data.size()[1], adv_data.size()[2], adv_data.size()[3]))
        #print("the shape of weight:", weight.size())
        for i in range(len(adv_data)):
            for j in range(10):
                output[i][j].backward(retain_graph=True)
                weight[i][j] = adv_data.grad[i].detach()
                adv_data.grad[i].data.zero_()

        #calculate the sigma
        reg_weight=np.zeros((len(adv_data),10,10))
        for i in range(len(adv_data)):
            weight_ = np.array(weight[i].reshape(10,-1).detach().cpu().numpy())
            norm = np.linalg.norm(x=weight_,ord=2,axis=1,keepdims=True)
            weight_n = weight_ / norm
            weight_n_trans =weight_n.transpose()
            #print("the shape of weight_0_trans:",weight_0_n_trans.shape)
            sigma_n = np.dot(weight_n,weight_n_trans)
            reg_weight[i]=sigma_n

        reg_weight=tensor2cuda(torch.from_numpy(np.asarray(reg_weight)).reshape(-1,100))
        #reg_weight=tensor2cuda(torch.from_numpy(np.asarray(reg_weight)))

        return reg_weight

    def pgd_whitebox(self, model, X, y, epsilon=8/255, num_steps=20, step_size=0.003):
        out = model(X)
        err = (out.data.max(1)[1] != y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)

        random_noise = tensor2cuda(torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon))
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = torch.optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
        return err, err_pgd


    def test_white_box(self, model, loader):
        # adv_test is False, return adv_acc as -1 
        model.eval()
        natural_err_total = 0.0
        robust_err_total = 0.0

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                #model.eval()
                X, y = Variable(data, requires_grad=True), Variable(label)
                err_natural, err_robust = self.pgd_whitebox(model, X, y)
                robust_err_total += err_robust
                natural_err_total += err_natural
        return 1 - natural_err_total / len(loader.dataset), 1- robust_err_total / len(loader.dataset)



    def test(self, model, loader, adv_test=False, use_pseudo_label=False):
        # adv_test is False, return adv_acc as -1 
        model.eval()
        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0
        
        feature=[[] for _ in range(10)]
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        with torch.no_grad():
            for data,label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                model.eval()
                output = model(data)
                for i in range(len(label)):
                    if len(feature[label[i]])<128:
                        feature[label[i]].append(output[i].detach().cpu().numpy())
        assert len(feature[0])==128,"size does not match"
        print("feature[0] shape:",np.array(feature[0]).shape)
        print("feature shape:",np.array(feature).shape)
        center=[]
        for i in range(len(feature)):
            feature_i=np.array(feature[i])
            #print(feature_i)
            center.append(np.mean(feature_i,axis=0))
        matrix=np.zeros((10,10))
        def l2loss(input1,input2):
            return np.mean(np.square(input1-input2))
        for i in range(len(center)):
            for j in range(len(center[0])):
                matrix[i][j]=l2loss(center[i],center[j])
        #print(matrix)
        def norm(matrix,min,max):
            matrix=(matrix-min)/(max-min)
            return matrix
        def norm2(matrix,max):
            matrix=matrix/max
            return matrix
        #matrix=norm(matrix,np.min(matrix),np.max(matrix))
        #matrix=norm2(matrix,np.max(matrix))
        new_matrix=[]
        for i in range(len(matrix)):
            matrix_i=norm2(matrix[i],np.max(matrix[i]))
            new_matrix.append(matrix_i)
        matrix=new_matrix
        #draw w * w^T
        from matplotlib import cm
        from matplotlib import axes
        fig=plt.figure()
        ax = fig.add_subplot(111)
        ax.set_yticks(range(len(matrix)))
        #ax.set_yticklabels(classes)
        ax.set_xticks(range(len(matrix[0])))
        #ax.set_xticklabels(classes)
        #作图并选择热图的颜色填充风格，这里选择hot
        im = ax.imshow(matrix, cmap=plt.cm.Blues)
        #增加右侧的颜色刻度条
        plt.colorbar(im)
        #增加标题
        plt.title("Feature Distance Matrix F")
        if not os.path.exists(args.adv+'_feature_distance'):
            os.mkdir(args.adv+'_feature_distance')
        save_path=os.path.join(args.adv+'_feature_distance','metrix.png')
        plt.savefig(save_path,dpi=900)
        plt.close()
        
        print("finish write the metrix1")
        return






def main(args):


    save_folder = '%s_%s_evaluate_resnet18_metrix' % (args.dataset, args.affix)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

    print_args(args, logger)

    model = ResNet18()

    attack = FastGradientSignUntargeted(model, args.epsilon, args.alpha, min_val=0, max_val=1,max_iters=args.k, _type=args.perturbation_type)
        
    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger, attack)

    if args.todo == 'train':
        transform_train = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                    (4,4,4,4), mode='constant', value=0).squeeze()),
                tv.transforms.ToPILImage(),
                tv.transforms.RandomCrop(32),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
            ])
        tr_dataset = tv.datasets.CIFAR10(args.data_root, 
                                       train=True, 
                                       transform=transform_train, 
                                       download=True)

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # evaluation during training
        te_dataset = tv.datasets.CIFAR10(args.data_root, 
                                       train=False, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


        trainer.train(model, model_linear, tr_loader, te_loader, args.adv_train)

    elif args.todo == 'test':
        te_dataset = tv.datasets.CIFAR10(args.data_root, 
                                       train=False, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        if args.adv_train:
            checkpoint = torch.load(args.load_adv_checkpoint)
        else:
            checkpoint = torch.load(args.load_std_checkpoint)
        model.load_state_dict(checkpoint)

        trainer.test(model, te_loader, adv_test=True, use_pseudo_label=False)

        #print("std acc: %.4f, adv_acc: %.4f" % (std_acc * 100, adv_acc * 100))

    else:
        raise NotImplementedError
    



if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)