import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import torchvision as tv

from time import time
import copy
from attack import FastGradientSignUntargeted
from torch.autograd import Variable
#from attack import OdiPgdWhiteBox
from model.resnet import * 
from model.resnet_linear_wbn import * 
#from model.resnet_linbp import *
from utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model

from argument_resnet_with_reg_weight import parser, print_args


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
                    # When training, the adversarial example is created from a random 
                    # close point to the original data point. If in evaluation mode, 
                    adv_data = self.attack.perturb(data, label, 'mean', True)

                    
                    model.train()
                    output=model(adv_data)

                else:
                    
                    
                    model.train()
                    
                    output = model(data)
                    
                #adv_loss = F.cross_entropy(output, label)
                class1_avg = (output[:,0]+output[:,1]+output[:,8]+output[:,9])/4
                class2_avg = (output[:,2]+output[:,3]+output[:,4]+output[:,5]+output[:,6]+output[:,7])/6
                reg_loss_1 = loss_fn(output[:,0],class1_avg)+loss_fn(output[:,1],class1_avg)+loss_fn(output[:,8],class1_avg)+loss_fn(output[:,9],class1_avg)
                reg_loss_2 = loss_fn(output[:,2],class2_avg)+loss_fn(output[:,3],class2_avg)+loss_fn(output[:,4],class2_avg)+loss_fn(output[:,5],class2_avg)+loss_fn(output[:,6],class2_avg)+loss_fn(output[:,7],class2_avg)
                reg_loss = (reg_loss_1+reg_loss_2) * 0.1
               
                

                #output2=model(data)
                adv_loss = F.cross_entropy(output, label)
                
                #print("reg_loss:",reg_loss)
                #print("adv_loss:",adv_loss)
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


    save_folder = '%s_%s_train_resnet18_with_reg_weight' % (args.dataset, args.affix)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

    print_args(args, logger)

    model = ResNet18(10)
    model_linear = ResNet18_linear_withoutbn(10)


    attack = FastGradientSignUntargeted(model, args.epsilon, args.alpha, min_val=0, max_val=1,max_iters=args.k, _type=args.perturbation_type)
        
    if torch.cuda.is_available():
        model.cuda()
        model_linear.cuda()

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

        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint)

        std_acc, adv_acc = trainer.test(model, te_loader, adv_test=True, use_pseudo_label=False)

        print("std acc: %.4f, adv_acc: %.4f" % (std_acc * 100, adv_acc * 100))

    else:
        raise NotImplementedError
    



if __name__ == '__main__':
    args = parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)