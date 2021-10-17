import numpy as np
import pickle
import torch
import os

with open('data/cifar-100-python/test','rb') as f:
    data = pickle.load(f, encoding='bytes')
print(data[b'data'].shape)#(50000,3072)
inputs = data[b'data'].reshape((-1,3,32,32)).transpose(0,2,3,1).astype(np.float32)
fine_labels = data[b'fine_labels'][:]
coarse_labels = data[b'coarse_labels'][:]
print(len(coarse_labels))#50000



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
#four super_class, each class 500 test

for i in range(len(coarse_labels)):
    if coarse_labels[i] == 1  or coarse_labels[i] == 2 or coarse_labels[i] == 5 or coarse_labels[i] == 14:

        
        
        if coarse_labels[i]==1:
            coarse_labels[i]=0
        if coarse_labels[i]==2:
            coarse_labels[i]=1
        if coarse_labels[i]==5:
            coarse_labels[i]=2
        if coarse_labels[i]==14:
            coarse_labels[i]=3
       


        index_=matrix.index(fine_labels[i])%5
        
        #print("index:",index)
        if index_<4:
            inputs2.append(inputs[i])
            coarse_labels2.append(coarse_labels[i])
            fine_labels2.append(index_+(coarse_labels[i])*4)
        else:
            #if fine_labels[i] == 67 or fine_labels[i]==73 or fine_labels[i]==91:
            inputs3.append(inputs[i])
            coarse_labels3.append(coarse_labels[i])
            fine_labels3.append(coarse_labels[i])

#print(fine_labels1)
print("val set length:",len(fine_labels2)) #800
print("test set length:",len(fine_labels3))

mask=np.unique(coarse_labels2) 
print("dif of val coarse2:",mask) #[1,2,5,14]
mask2=np.unique(fine_labels2)
print("dif of test fine2:",mask2)

mask=np.unique(coarse_labels3) 
print("dif of val coarse3:",mask)
mask2=np.unique(fine_labels3)
print("dif of test fine3:",mask2)

save_dir='data/cifar20'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



with open(os.path.join(save_dir,'test_400'), 'wb') as fout:
    pickle.dump({'data': inputs2, 'labels': fine_labels2}, fout)

with open(os.path.join(save_dir,'test_100'), 'wb') as fout:
    pickle.dump({'data': inputs3, 'labels': fine_labels3}, fout)


with open(os.path.join(save_dir,'test_400_coarse'), 'wb') as fout:
    pickle.dump({'data': inputs2, 'labels': coarse_labels2}, fout)

