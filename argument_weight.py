import argparse

def parser():
    parser = argparse.ArgumentParser(description='Video Summarization')
    parser.add_argument('--todo', choices=['train', 'valid', 'test', 'visualize'], default='test',
        help='what behavior want to do: train | valid | test | visualize')
    parser.add_argument('--dataset', default='cifar-10', help='use what dataset')
    parser.add_argument('--data_root', default='/data', 
        help='the directory to save the dataset')
    parser.add_argument('--log_root', default='log_weight', 
        help='the directory to save the logs or other imformations (e.g. images)')
    parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
    parser.add_argument('--checkpoint_std', default='checkpoint/cifar-10_std_train_baseline/last_model.pth')
    parser.add_argument('--checkpoint_adv', default='checkpoint/cifar-10_adv_train_baseline/last_model.pth')
    parser.add_argument('--adv', choices=['adv','std','trades','adv_svhn','std_svhn','std_cifar100','adv_cifar100','std_stl','adv_stl','std_tinyimagenet','adv_tinyimagenet','std_cifar20','adv_cifar20','std_cifar40','adv_cifar40'],default='adv',help='what type of model to do: adv | std | trades')
    parser.add_argument('--affix', default='pgd20', help='the affix for the save folder')

    # parameters for generating adversarial examples
    parser.add_argument('--epsilon', '-e', type=float, default=8/255, 
        help='maximum perturbation of adversaries (8/255=0.0314)')
    parser.add_argument('--alpha', '-a', type=float, default=0.007, 
        help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
    parser.add_argument('--k', '-k', type=int, default=10, 
        help='maximum iteration when generating adversarial examples')
    parser.add_argument('--attack_method', choices=['PGD', 'FGSM'], default='PGD',
        help='Attack type')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
    parser.add_argument('--max_epoch', '-m_e', type=int, default=200, 
        help='the maximum numbers of the model see a sample')

    parser.add_argument('--gpu', '-g', default='2', help='which gpu to use')
    parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf', 
        help='the type of the perturbation (linf or l2)')
    
    parser.add_argument('--read_layer_weight', action='store_true')
    parser.add_argument('--evaluate_weight', action='store_true')
    parser.add_argument('--adv_model', action='store_true')


    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))