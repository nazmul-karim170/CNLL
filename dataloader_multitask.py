from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os
from autoaugment import CIFAR10Policy, ImageNetPolicy
from torchnet.meter import AUCMeter
import torch.nn.functional as F 
from sklearn.metrics import confusion_matrix
from Asymmetric_Noise import *


## For plotting the logs
import wandb
wandb.init(project="noisy-label-project", entity="ryota170")


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

transform_none_10 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_mnist_10 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.), (0.2023, 0.1994, 0.2010)),
    ]
)


transform_none_100 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ]
)

transform_weak_10 = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_weak_100 = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ]
)


transform_strong_10 = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


transform_strong_100 = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ]
)

class cifar_dataset(Dataset): 
    def __init__(self, dataset, task_mode, sample_ratio, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], log=''): 
        
        self.r = r # noise ratio
        self.sample_ratio = sample_ratio
        self.transform = transform
        self.mode = mode
        self.noise_mode = noise_mode

        self.class_ind = {}

        if self.noise_mode=='rand':
            ext = 'rand_' + str(r) 
        else:
            ext = '' 

        if self.mode=='test':
            if dataset=='cifar10':
                # for jj in range(2):
                #     num = 'task_' + str(jj)                 
                feature_file = root_dir + '/Cifar10_test_images_' + str(task_mode) + '.npy'
                label_file   = root_dir + '/Cifar10_test_labels_' + str(task_mode) + '.npy'
                self.test_data  = np.squeeze(np.load(feature_file))
                self.test_data  = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = np.squeeze(np.load(label_file))

            elif dataset=='cifar100':
                feature_file = root_dir + '/Cifar100_test_images_' + ext+ str(task_mode) + '.npy'
                label_file   = root_dir + '/Cifar100_test_labels_' + ext+ str(task_mode) + '.npy'
                self.test_data  = np.squeeze(np.load(feature_file))
                self.test_data  = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = np.squeeze(np.load(label_file))           

        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                feature_file =   root_dir + '/Cifar10_train_images_' + str(task_mode) + '_' + str(self.noise_mode) + '_' + str(self.r) + '.npy'
                label_file   =   root_dir + '/Cifar10_train_labels_' + str(task_mode) + '_' + str(self.noise_mode) + '_' + str(self.r) + '.npy'
                clean = 1-np.squeeze(np.load(root_dir + '/Cifar10_noise_labels_'+ str(task_mode) + '_'  + str(self.noise_mode) + '_' + str(self.r)+ '.npy'))
                train_data  = np.squeeze(np.load(feature_file))
                train_label = np.squeeze(np.load(label_file))
                num_class_sample = np.shape(train_label)[0]
            
            elif dataset=='cifar100':
                feature_file =   root_dir + '/Cifar100_train_images_' + str(task_mode) + '_' + str(self.noise_mode) + '_' + str(self.r) + '.npy'
                label_file   =   root_dir + '/Cifar100_train_labels_'+ str(task_mode) + '_'  + str(self.noise_mode) + '_' + str(self.r)+ '.npy'
                clean = 1-np.squeeze(np.load(root_dir + '/Cifar100_noise_labels_'+ str(task_mode) + '_'  + str(self.noise_mode) + '_' + str(self.r)+ '.npy'))
                
                train_data  = np.squeeze(np.load(feature_file))
                train_label = np.squeeze(np.load(label_file))
                num_class_sample = np.shape(train_label)[0]

                ### Get number of classes and fix the orientation of dimension ###
            num_class  = np.shape(np.unique(train_label))[0]
            train_data = train_data.reshape((num_class_sample, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            num_sample = num_class_sample    
            noise_label = train_label

            for kk in range(num_class):
                self.class_ind[kk] = [i for i,x in enumerate(noise_label) if x==kk]

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
                
            else:
                save_image_file     = root_dir  + '/Clean_images_' + str(dataset) + '_' + str(task_mode) + '_' + str(noise_mode) +'_' + str(self.r) + '.npy'
                save_label_file     = root_dir  + '/Clean_labels_' + str(dataset) + '_' + str(task_mode) + '_' + str(noise_mode) +'_' + str(self.r) + '.npy'
                save_unlabeled_file = root_dir  + '/Clean_unlabeled_' + str(dataset) + '_' + str(task_mode) + '_' + str(noise_mode) +'_' + str(self.r) + '.npy'

                image_file     = os.path.join( save_image_file)
                label_file     = os.path.join( save_label_file)
                unlabeled_file = os.path.join(save_unlabeled_file)

                if self.mode == "labeled":
                    pred_idx = np.argsort(probability.cpu().numpy())[0: int(self.sample_ratio*num_sample)]
                    
                    pred_idx = [int(x) for x in list(pred_idx)]
                    idx = list(range(num_sample))
                    pred_idx_noisy = [x for x in idx if x not in pred_idx]
                    IND1 = [x for x in pred_idx_noisy if x in pred_idx]
                    np.save(save_image_file, train_data[pred_idx])
                    np.save(save_label_file, noise_label[pred_idx])
                    np.save(save_unlabeled_file, train_data[pred_idx_noisy])
                    probability[probability<0.5] = 0
                    self.probability = [1-probability[i] for i in pred_idx]

                                    
                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                                 

    def num_samples(self):
        return self.num_sample

    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = 255*self.train_data[index], self.noise_label[index], self.probability[index]
            img = img.astype(np.uint8)
            image = Image.fromarray(img)
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)

            return img1, img2, img3, img4, target, prob   

        elif self.mode=='unlabeled':
            img = 255*self.train_data[index]
            img = img.astype(np.uint8)
            image = Image.fromarray(img)
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            return img1, img2, img3, img4

        elif self.mode=='all':
            img, target = 255*self.train_data[index], self.noise_label[index]
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index

        elif self.mode=='test':
            img, target = 255*self.test_data[index], self.test_label[index]
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)



class cifar_dataloader():  
    def __init__(self, dataset, task_mode, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.task_mode = task_mode
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file

        if self.dataset=='cifar10':
            self.transforms = {
                "warmup": transform_weak_10,
                "unlabeled": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
                "labeled": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
            }

            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])

        elif self.dataset=='cifar100':
            self.transforms = {
                "warmup": transform_weak_100,
                "unlabeled": [
                            transform_weak_100,
                            transform_weak_100,
                            transform_strong_100,
                            transform_strong_100
                        ],
                "labeled": [
                            transform_weak_100,
                            transform_weak_100,
                            transform_strong_100,
                            transform_strong_100
                        ],
            }        
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
                   
    def run(self,sample_ratio, mode, class_name=[], pred=[], prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, task_mode= self.task_mode, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["warmup"], mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader                      

        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, task_mode= self.task_mode, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["labeled"], mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, drop_last=True)  

            return labeled_trainloader          
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, task_mode= self.task_mode, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, task_mode= self.task_mode, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers, drop_last= True)          
            return eval_loader        