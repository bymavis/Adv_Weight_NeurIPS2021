import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import torchvision as tv

from time import time

from attack import FastGradientSignUntargeted
from torch.autograd import Variable
#from attack import OdiPgdWhiteBox
#from model.resnet import * 
from model.resnet import * 
from utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model
import argparse
from mydata import *

#torch.cuda.set_device(6)
parser = argparse.ArgumentParser(description='Video Summarization')
parser.add_argument('--todo', choices=['train', 'valid', 'test', 'visualize'], default='train',
                    help='what behavior want to do: train | valid | test | visualize')
parser.add_argument('--dataset', default='cifar-100', help='use what dataset')
parser.add_argument('--data_root', default='data/cifar20',
                    help='the directory to save the dataset')
parser.add_argument('--log_root', default='log',
                    help='the directory to save the logs or other imformations (e.g. images)')
parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
parser.add_argument('--load_checkpoint', default='./model/default/model.pth')
parser.add_argument('--affix', default='default', help='the affix for the save folder')

# parameters for generating adversarial examples
parser.add_argument('--epsilon', '-e', type=float, default=2.0/255,
                    help='maximum perturbation of adversaries (4/255=0.0157)')
parser.add_argument('--alpha', '-a', type=float, default=1.0/255,
                    help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
parser.add_argument('--k', '-k', type=int, default=5,
                    help='maximum iteration when generating adversarial examples')

parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
parser.add_argument('--max_epoch', '-m_e', type=int, default=200,
                    help='the maximum numbers of the model see a sample')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-1, help='learning rate')
parser.add_argument('--weight_decay', '-w', type=float, default=2e-4,
                    help='the parameter of l2 restriction for weights')

parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')
parser.add_argument('--n_eval_step', type=int, default=100,
                    help='number of iteration per one evaluation')
parser.add_argument('--n_checkpoint_step', type=int, default=400,
                    help='number of iteration to save a checkpoint')
parser.add_argument('--n_store_image_step', type=int, default=400,
                    help='number of iteration to save adversaries')
parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf',
                    help='the type of the perturbation (linf or l2)')

parser.add_argument('--adv_train', action='store_true')

parser.add_argument('--beta', '-beta', type=float, default=1.,
                    help='channel regularization')

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

    def standard_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, False)

    def adversarial_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, True)

    def train(self, model, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger

        #opt = torch.optim.SGD(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
        opt = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, 
                                                         milestones=[75, 90], 
                                                         gamma=0.1)
        _iter = 0
       
        begin_time = time()
        best_pred = 0.0
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        for epoch in range(1, args.max_epoch+1):
            model.train()
            scheduler.step()
            for data, label in tr_loader:
                model.train()
                data, label = tensor2cuda(data), tensor2cuda(label)                       
                
                if adv_train:
                    # When training, the adversarial example is created from a random 
                    # close point to the original data point. If in evaluation mode, 
                    adv_data = self.attack.perturb(data, label, 'mean', True)

                    model.train()
                    output=model(adv_data)

                else:
                    output = model(data)
                
                class1_avg=((output[:,0]+output[:,1]+output[:,2]+output[:,3])/4).detach()
                class2_avg=((output[:,4]+output[:,5]+output[:,6]+output[:,7])/4).detach()
                class3_avg=((output[:,8]+output[:,9]+output[:,10]+output[:,11])/4).detach()
                class4_avg=((output[:,12]+output[:,13]+output[:,14]+output[:,15])/4).detach()
                
                
                reg_loss1=(loss_fn(output[:,0],class1_avg)+loss_fn(output[:,1],class1_avg)+loss_fn(output[:,2],class1_avg)+loss_fn(output[:,3],class1_avg))/4
                reg_loss2=(loss_fn(output[:,4],class2_avg)+loss_fn(output[:,5],class2_avg)+loss_fn(output[:,6],class2_avg)+loss_fn(output[:,7],class2_avg))/4
                reg_loss3=(loss_fn(output[:,8],class3_avg)+loss_fn(output[:,9],class3_avg)+loss_fn(output[:,10],class3_avg)+loss_fn(output[:,11],class3_avg))/4
                reg_loss4=(loss_fn(output[:,12],class4_avg)+loss_fn(output[:,13],class4_avg)+loss_fn(output[:,14],class4_avg)+loss_fn(output[:,15],class4_avg))/4
                
                reg_loss=0.25*reg_loss1+0.25*reg_loss2+0.25*reg_loss3+0.25*reg_loss4
                adv_loss = F.cross_entropy(output, label)

                loss=args.beta*reg_loss+adv_loss
                #loss = F.cross_entropy(output, label)
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
                new_pred = va_acc_20
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

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                model.eval()
                output = model(data)

                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                
                total_acc += te_acc
                num += output.shape[0]

                if adv_test:
                    # use predicted label as target label
                    with torch.enable_grad():
                        adv_data = self.attack.perturb(data, 
                                                       pred if use_pseudo_label else label, 
                                                       'mean', 
                                                       True)
                    model.eval()
                    adv_output = model(adv_data)

                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                else:
                    total_adv_acc = -num

        return total_acc / num , total_adv_acc / num

def main(args):
    save_folder = '%s_train_resnet18_cluster' % (args.affix)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

    print_args(args, logger)


    model = ResNet18(num_classes=16)

    attack =FastGradientSignUntargeted(model,
                                        args.epsilon,
                                        args.alpha,
                                        min_val=0,
                                        max_val=1,
                                        max_iters=args.k,
                                        _type=args.perturbation_type)

    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger, attack)

    


    train_root = 'data/cifar20/train_400'
    test_root = 'data/cifar20/test_400'    
        
    if args.todo == 'train':
        transform_train = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                 (4, 4, 4, 4), mode='constant', value=0).squeeze()),
            tv.transforms.ToPILImage(),
            tv.transforms.RandomCrop(32),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
        ])

        tr_dataset = MyDataset(train_root, transform=transform_train)

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


        te_dataset = MyDataset(test_root, transform=transform_train)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainer.train(model, tr_loader, te_loader, args.adv_train)
    elif args.todo == 'test':

        te_dataset = MyDataset(test_root, transform=tv.transforms.ToTensor())

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint)

        std_acc, adv_acc = trainer.test(model, te_loader, adv_test=True, use_pseudo_label=False)

        print("std acc: %.4f, adv_acc: %.4f" % (std_acc * 100, adv_acc * 100))

    else:
        raise NotImplementedError


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)