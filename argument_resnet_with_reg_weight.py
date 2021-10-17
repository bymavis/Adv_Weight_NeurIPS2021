import argparse

def parser():
    parser = argparse.ArgumentParser(description='Video Summarization')
    parser.add_argument('--todo', choices=['train', 'valid', 'test', 'visualize'], default='train',
        help='what behavior want to do: train | valid | test | visualize')
    parser.add_argument('--dataset', default='cifar-10', help='use what dataset')
    parser.add_argument('--data_root', default='/data', 
        help='the directory to save the dataset')
    parser.add_argument('--log_root', default='log', 
        help='the directory to save the logs or other imformations (e.g. images)')
    parser.add_argument('--checkpoint_std', default='checkpoint/cifar-10_linf_train_std_resnet18/checkpoint_76000.pth')
    parser.add_argument('--checkpoint_adv', default='checkpoint/cifar-10_linf_adv_loss_baseline/last_model.pth')
    parser.add_argument('--adv', choices=['adv','std','trades'],default='std',help='what type of model to do: adv | std | trades')
    parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
    parser.add_argument('--load_checkpoint', default='checkpoint/cifar-10_linf_adv_loss_baseline/checkpoint_40000.pth')
    parser.add_argument('--affix', default='pgd20', help='the affix for the save folder')

    # parameters for generating adversarial examples
    parser.add_argument('--epsilon', '-e', type=float, default=8/255, 
        help='maximum perturbation of adversaries (8/255=0.0314)')
    parser.add_argument('--alpha', '-a', type=float, default=0.007, 
        help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
    parser.add_argument('--beta', type=float, default=0.1, 
        help='the weight of regularization adv loss')
    parser.add_argument('--beta_cos', type=float, default=1.0, 
        help='the weight of regularization  cos loss')
    parser.add_argument('--k', '-k', type=int, default=10, 
        help='maximum iteration when generating adversarial examples')
    parser.add_argument('--attack_method', choices=['PGD', 'FGSM'], default='PGD',
        help='Attack type')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
    parser.add_argument('--max_epoch', '-m_e', type=int, default=200, 
        help='the maximum numbers of the model see a sample')
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
    parser.add_argument('--weight_norm', action='store_true')
    parser.add_argument('--weight_var', action='store_true')
    parser.add_argument('--weight_cluster', action='store_true')


    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))