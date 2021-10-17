### Adv_Weight_NeurIPS2021
The officially code (pytorch version) for paper 'Clustering Effect of Adversarial Robust Models'. (accepted by NeurIPS 2021, Spotlight)
### Usage

compute correlation matrix for ResNet18 
```
#Robust Model
python evaluate_weight_correlation.py --adv adv --evaluate_weight
#Non-Robust Model
python evaluate_weight_correlation.py --adv std --evaluate_weight
```

compute feature distance  for ResNet18
```
#Robust Model
python evaluate_feature_distance.py --adv adv --adv_train
#Non-Robust Model
python evaluate_feature_distance.py --adv std
```

train robust model with an enhanced clustering effect for ResNet18 on CIFAR-10
```
python train_resnet18_cifar10_cluster.py --adv_train --affix cluster --beta 0.1
```

reconstruct new CIFAR-20 data set,take five superclasses and four subclasses as an example 
```
python read_cifar20_train.py
python read_cifar20_test.py
```

train robust model with an enhanced clustering effect for ResNet18 on new data set
```
python train_resnet18_cifar20_cluster.py --adv_train --affix adv_beta01 --beta 1
```

finetune robust model with an enhanced clustering effect for ResNet18 on new data set
```
python train_resnet18_cifar_20_finetune_fc_cls.py --affix cls --adv cls --beta 2
```

test cifar-20 R+C models
```
python test_resnet18_cifar_20.py --gpu 2 --affix cls --cls_train
```
