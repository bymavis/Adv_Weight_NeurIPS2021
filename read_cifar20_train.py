import numpy as np
import pickle
import torch
import os

with open('data/cifar-100-python/train','rb') as f:
    data = pickle.load(f, encoding='bytes')
print(data[b'data'].shape) #(10000,3072)
inputs = data[b'data'].reshape((-1,3,32,32)).transpose(0,2,3,1).astype(np.float32)
fine_labels = data[b'fine_labels'][:]
coarse_labels = data[b'coarse_labels'][:]
print("coarse_labels length:",len(coarse_labels)) #10000


inputs1 = []
fine_labels1 = []
coarse_labels1 = []
  
inputs2 = []
fine_labels2 = []
coarse_labels2 = []

inputs3 = []
fine_labels3 = []
coarse_labels3 = []
 
matrix = [4,30,55,72,95,
1,32,67,73,91,
54,62,70,82,92,
9,10,16,28,61,
0,51,53,57,83,
22,39,40,86,87,
5,20,25,84,94,
6,7,14,18,24,
3,42,43,88,97,
12,17,37,68,76,
23,33,49,60,71,
15,19,21,31,38,
34,63,64,66,75,
26,45,77,79,99,
2,11,35,46,98,
27,29,44,78,93,
36,50,65,74,80,
47,52,56,59,96,
8,13,48,58,90,
41,69,81,85,89] 
#four super_class, each class 500 tes

for i in range(len(coarse_labels)):

    if coarse_labels[i] == 1  or coarse_labels[i] == 2 or coarse_labels[i] == 5 or coarse_labels[i] == 14:

        index_=matrix.index(fine_labels[i])%5
        
        if coarse_labels[i]==1:
            coarse_labels[i]=0
        if coarse_labels[i]==2:
            coarse_labels[i]=1
        if coarse_labels[i]==5:
            coarse_labels[i]=2
        if coarse_labels[i]==14:
            coarse_labels[i]=3
        
        if index_<4:
            inputs1.append(inputs[i])
            coarse_labels1.append(coarse_labels[i])
            fine_labels1.append(index_+(coarse_labels[i])*4)
        else:
            inputs2.append(inputs[i])
            coarse_labels2.append(coarse_labels[i])
            fine_labels2.append(coarse_labels[i])
#print 4000
print("train set length:",len(fine_labels1))#2000
mask=np.unique(coarse_labels1) 
print("dif of coarse:",mask) #[0,1,2,3]
mask2=np.unique(fine_labels1)
print("dif of fine:",mask2)

print("train set length:",len(coarse_labels2))#2000
mask=np.unique(coarse_labels2) 
print("dif of coarse:",mask) #[0,1,2,3]
mask2=np.unique(fine_labels2)
print("dif of fine:",mask2)





save_dir='data/cifar20'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(os.path.join(save_dir,'train_400'), 'wb') as fout:
    pickle.dump({'data': inputs1, 'labels': fine_labels1}, fout)

with open(os.path.join(save_dir,'train_test_100'), 'wb') as fout:
    pickle.dump({'data': inputs2, 'labels': fine_labels2}, fout)


with open(os.path.join(save_dir,'train_400_coarse'), 'wb') as fout:
    pickle.dump({'data': inputs1, 'labels': coarse_labels1}, fout)

