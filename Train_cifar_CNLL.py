from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import random
import os
import argparse
import numpy as np
from PreResNet_multitask import *
from dataloader_separation import *
from math import log2
from Contrastive_loss import *

import collections.abc
from collections.abc import MutableMapping


## For plotting the logs
import wandb
wandb.init(project="continual-noisy-label-project", entity="ryota170")   ## Change the entity here (your 'Weights and Biases' username)

## Arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--warm_up', default=5, type=int, help='warmup epochs') 
parser.add_argument('--lr', '--learning_rate', default=0.0005, type=float, help='initial learning rate')    ### Learning Rate Should not be more than 0.001
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=30, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--finetune_epochs', default=10, type=int)
parser.add_argument('--task_set',  default='vehicles10-large_animals10-reset80', choices=['vehicles10-manmade_objects15-reset75', 'vehicles10-manmade_objects10-reset80', 'vehicles10-large_animals10-reset80'])
parser.add_argument('--task_mode', default="task_0", type=str, help="Which task we are executing")
parser.add_argument('--r', default=0.2, type=float, help='noise ratio')
parser.add_argument('--tau', default=5, type=float, help='filtering coefficient')
parser.add_argument('--metric', type=str, default = 'JSD', help='Comparison Metric')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--resume', default=False, type=bool, help = 'Resume from the warmup checkpoint')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./data/CIFAR10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
args = parser.parse_args()

## Weigths and Biases Configuration
wandb.config = {
  "Learning Rate": args.lr,
  "warmup_epochs": args.warmup_epochs,
  "finetune_epochs": args.finetune_epochs,
  "Batch Size": args.batch_size,
  "Dataset": args.dataset,
  "Noise Mode": args.noise_mode,
  "Noise Rate": args.r,
  "Loss Metric": args.metric
}

## GPU Setup 
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

        ##  Download the Datasets  ##
if args.dataset== 'cifar10':
    torchvision.datasets.CIFAR10(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR10(args.data_path,train=False, download=True)
else:
    torchvision.datasets.CIFAR100(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR100(args.data_path,train=False, download=True)

        ## Checkpoint Location ##
folder = args.dataset + '_' + args.noise_mode + '_' +  str(args.metric) + '_' + str(args.r) 
model_save_loc1 = './checkpoint/' + folder
if not os.path.exists(model_save_loc1):
    os.mkdir(model_save_loc1)

        ## Log files ##
stats_log=open(model_save_loc1 +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open(model_save_loc1 +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')     
test_loss_log = open(model_save_loc1 +'/test_loss.txt','w')

        ## SSL-Training ##
def train(epoch, net, optimizer, labeled_trainloader, unlabeled_trainloader):
    # net2.eval()               # Freeze one network and train the other
    net.train()

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1

    ## Loss Statistics
    loss_x = 0
    loss_u = 0
    loss_scl = 0
    loss_ucl = 0

    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
        
        batch_size = inputs_x.size(0)

        # Transform Label to One-hot
        labels_x = torch.zeros(batch_size, task_classes).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()
        
        with torch.no_grad():
            # Label Co-guessing of Unlabeled Samples
            outputs_u11 = net(inputs_u)[1]
            outputs_u12 = net(inputs_u2)[1]
       
            ## Pseudo-Label
            pu  = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1))/2
            ptu = pu**(1/args.T)            ## Temparature Sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()                  

            ## Label Refinement
            outputs_x  = net(inputs_x)[1]
            outputs_x2 = net(inputs_x2)[1]            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T)    # Temparature sharpening 
                        
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)           
            targets_x = targets_x.detach()

        # MixMatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
        all_inputs  = torch.cat([inputs_x, inputs_x2, inputs_u3, inputs_u4], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b   = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        ## Mixup
        mixed_input  = l * input_a  + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        _ , logits1, logits = net(mixed_input)
        logits_x = logits1[:batch_size*2]
        logits_x1 = logits[:batch_size*2]

        logits_u = logits[batch_size*2:]        
        
        ## Combined Loss
        Lx, Lu, Ldiv, lamb = criterion(logits_x, logits_x1, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # Regularization
        prior = torch.ones(task_classes)/task_classes
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        ## Total Loss
        loss = Lx + 0.1*lamb * Lu + penalty

        ## Accumulate Loss
        loss_x += Lx.item()
        loss_u += Lu.item()
        loss_ucl += Ldiv.item()

        # Compute Gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f Contrastive Loss:%.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.finetune_epochs, batch_idx+1, num_iter, loss_x/(batch_idx+1), loss_u/(batch_idx+1),  loss_ucl/(batch_idx+1)))
        sys.stdout.flush()

    wandb.log({"Total Loss":  0,
               "Labeled Loss": loss_x/(batch_idx+1),
               "KL Div. Loss":  loss_ucl/(batch_idx+1),
               "Unlabeled Loss": loss_u/(batch_idx+1)})


## For Standard Training 
def warmup_standard(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    total = 0
    correct = 0

    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        _,outputs, _ = net(inputs)
               
        loss         = CEloss(outputs, labels)
        _, predicted = torch.max(outputs, 1)

        ## Penalize confident prediction for asymmetric noise                    
        if args.noise_mode=='asym':         
            penalty = conf_penalty(outputs)
            if torch.isnan(penalty):
                L = loss
            else:
                L = loss + penalty      
        else:   
            L = loss

        L.backward()  
        optimizer.step()                
        total   += labels.size(0)
        correct += predicted.eq(labels).cpu().sum().item()

        # sys.stdout.write('\r')
        # sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
        #         %(args.dataset, args.r, args.noise_mode, epoch, args.warmup_epochs, batch_idx+1, num_iter, loss.item()))
        # sys.stdout.flush()
    
    acc = 100.*correct/total
    # print("\n| Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch, acc))  


## For Standard Training 
def warmup_val(epoch,net,optimizer,dataloader):

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    total = 0
    correct = 0

    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        _, outputs, _  = net(inputs)               
        _, predicted = torch.max(outputs, 1)                          

        total   += labels.size(0)
        correct += predicted.eq(labels).cpu().sum().item()

    acc = 100.*correct/total
    print("\n| Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch, acc))  
    return acc

## Test Accuracy
def warmup_test(epoch,net1, class_name):
    net1.eval()

    num_samples = 1000
    correct = 0
    total   = 0
    loss_x  = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1, outputs2 = net1(inputs)
            # _, outputs2 = net2(inputs)           
            outputs = outputs1

            _, predicted = torch.max(outputs, 1)            
            loss = CEloss(outputs, targets)  
            loss_x  += loss.item()
            total   += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()  

    acc = 100.*correct/total
    # print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write(str(acc)+'\n')
    test_log.flush()  
    test_loss_log.write(str(loss_x/(batch_idx+1))+'\n')
    test_loss_log.flush()
    return acc, loss_x/(batch_idx+1)


## Test Accuracy
def test(epoch,net1):
    net1.eval()
    # net2.eval()

    correct = 0
    total = 0
    loss_x = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1, outputs2 = net1(inputs)
            # _, outputs2 = net2(inputs)           
            outputs = outputs1 

            _, predicted = torch.max(outputs, 1)            
            loss = CEloss(outputs, targets)  
            loss_x  += loss.item()

            total   += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()  

    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy (Whole Test Dataset): %.2f%%\n" %(epoch,acc))  
    test_log.write(str(acc)+'\n')
    test_log.flush()  
    test_loss_log.write(str(loss_x/(batch_idx+1))+'\n')
    test_loss_log.flush()
    return acc, loss_x/(batch_idx+1)


## KL Divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self,p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

## Uniform Sample Selection JSD based 
def sample_selection_JSD(epoch, model1, num_samples, class_name):  
    JS_dist = Jensen_Shannon()
    JSD   = torch.zeros(num_samples)    

    for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])     
            out2 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[2])

        out = torch.zeros(out1.size()).cuda()
        out[:, class_name] = out1[:, class_name]

        _, ind = torch.max(out, 1)
        out_final = torch.zeros(out1.size()).cuda()
        for kk in range(out.size()[0]):
            out_final[kk, ind[kk]] = 1
            
                        ## make one hot prediction ## 
            # if out[kk,class_name[0]]> out[kk,class_name[1]]:
            #     out[kk,class_name[0]] = 1
            #     out[kk,class_name[1]] = 0
            # else:
            #     out[kk,class_name[0]] = 0
            #     out[kk,class_name[1]] = 1
            # print(ind[kk],out_final[kk,0:ind[kk]+1])

        ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
        dist = JS_dist(out_final,  F.one_hot(targets, num_classes = args.num_class))
        JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist

    return JSD

## Unsupervised Loss coefficient adjustment 
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, outputs_x2, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        ## Get the labeled and unlabeled probability distributions
        labeled_distribution   = F.softmax(outputs_x,dim=1)
        unlabeled_distribution = F.log_softmax(outputs_x2,dim=1)

        L_div = F.kl_div(unlabeled_distribution, labeled_distribution, reduction='batchmean')
        
        return Lx, Lu, L_div, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model(num_class):
    model = ResNet18(num_classes = num_class)
    model = model.cuda()
    return model

      ### Model Specifications and Loss Functions ###
print('| Building net')
task_classes = args.num_class
net1 = create_model(task_classes)
cudnn.benchmark = True

## Semi-Supervised Loss
criterion  = SemiLoss()
contrastive_criterion = SupConLoss()

## Optimizer and Scheduler
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 240, 2e-4)
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

## Warmup Period
if args.dataset =='cifar10':
    warm_up = args.warm_up
elif args.dataset=='cifar100':
    warm_up = args.warm_up

num_samples = 0


## Checkpoint Location
folder = args.dataset + '_' + str(args.noise_mode) + '_' + str(args.r)
model_save_loc = './checkpoint/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)
model_name_1 = 'Net1_' + str(args.task_mode) +'.pth'
if args.resume:
    start_epoch = warm_up
    net1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1))['net'])
else:
    start_epoch = 0

if args.dataset=='cifar10':
    task_mode_list = ['task_0', 'task_1', 'task_2', 'task_3', 'task_4']
else:
    task_mode_list = ['task_0', 'task_1', 'task_2', 'task_3', 'task_4', 'task_5', 'task_6', 'task_7', 'task_8', 'task_9', 'task_10', 'task_11', 'task_12', 'task_13', 'task_14', 'task_15', 'task_16', 'task_17', 'task_18', 'task_19']

task_mode_finetune = []

## Intitalize Replay Buffers
clean_rep_buffer = {"Images": [],
                "Labels": []}
noisy_rep_buffer = {"Images": []}
N_1 = 25
N_2 = 50


for task_mode in task_mode_list:


    if args.dataset=='cifar10':
        label_file  = args.data_path + '/cifar10_Train_labels_' + str(task_mode) + '_' + str(args.noise_mode) + '_' + str(args.r) + '.npy'   
    else:
        label_file  = args.data_path + '/cifar100_Train_labels_' + str(task_mode) + '_' + str(args.noise_mode) + '_' + str(args.r) + '.npy'   

    train_label = np.squeeze(np.load(label_file))
    class_name  = np.unique(train_label)
    num_samples = np.shape(train_label)[0]
    print("Number of Samples:", num_samples, class_name)

    ## Create Model
    weight = torch.zeros(task_classes)
    weight[class_name] = 1
    weight = weight.cuda()

    ## Weighted Loss Functions
    CEloss   = nn.CrossEntropyLoss(weight = weight)

    ## Intialize Clean and Noisy Buffers
    delay_buffer_size = 500
    clean_buffer_size = 500
    noisy_buffer_size = 1000

    clean_buffer = None
    noisy_buffer = None
    clean_buffer = {"Images": [],
                    "Labels": []}
    noisy_buffer = {"Images": []}

    nb_iterations = int(num_samples/delay_buffer_size)
    for iteration in range(nb_iterations):
        best_acc = 0
        ## Call the Dataloader
        from dataloader_separation import *
        loader = cifar_dataloader(args.dataset, task_mode=task_mode, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size, num_workers=4,\
            root_dir=args.data_path, log=stats_log, noise_file='')

        eval_loader = loader.run(0, [iteration, iteration+1], delay_buffer_size, 'eval_train') 
        warmup_dataset, warmup_trainloader = loader.run(0, [iteration, iteration+1], delay_buffer_size,  'warmup')
        test_loader = loader.run(0, [iteration, iteration+1], delay_buffer_size, 'test', list(class_name))

        ## Warmup Training 
        for epoch in range(0,args.warmup_epochs):
              
            ## Warmup Stage 
            # warmup_trainloader = loader.run(0, delay_buffer_size, 'warmup')
            
            print('Warmup Model')
            warmup_standard(epoch, net1, optimizer1, warmup_trainloader)   
            acc, loss = warmup_test(epoch, net1, class_name)
            scheduler1.step()

            ## Keep the log
            wandb.log({"Validation Accuracy": acc, 
                       "Validation Loss": loss})
            wandb.watch(net1)

            if acc > best_acc:
                model_name_1 = 'Net1_' + str(args.task_mode) +  '.pth'
                print("Save the Model --- --")

                checkpoint1 = {
                    'net': net1.state_dict(),
                    'Model_number': 1,
                    'Noise_Ratio': args.r,
                    'Loss Function': 'CE',
                    'Optimizer': 'SGD',
                    'Noise_mode': args.noise_mode,
                    'Accuracy': acc,
                    'Dataset': args.dataset,
                    'Batch Size': args.batch_size,
                    'epoch': epoch,
                }
                torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
                best_acc = acc
        
                        ## Separate the Delay Buffer
        JSD = sample_selection_JSD(0, net1, delay_buffer_size, class_name)  
        threshold = torch.mean(JSD)
        SR = torch.sum(JSD<threshold).item()/delay_buffer_size
        # print("Threshold:", threshold, SR)

        JSD = JSD.cpu().numpy()
        pred_idx = np.argsort(JSD)[0: int(SR*delay_buffer_size)]
        pred_idx = [int(x) for x in list(pred_idx)]
        idx = list(range(delay_buffer_size))
        pred_idx_noisy = [x for x in idx if x not in pred_idx]
        IND1 = [x for x in pred_idx_noisy if x in pred_idx]

        clean_buffer["Images"].extend(warmup_dataset.train_data[pred_idx])
        clean_buffer["Labels"].extend(warmup_dataset.noise_label[pred_idx])
        noisy_buffer["Images"].extend(warmup_dataset.train_data[pred_idx_noisy])

        ## Replay Buffer
        repl_idx       = np.array(pred_idx)[:N_1]
        repl_idx_noisy = np.array(pred_idx_noisy)[:N_2]
        clean_rep_buffer["Images"].extend(warmup_dataset.train_data[repl_idx])
        clean_rep_buffer["Labels"].extend(warmup_dataset.noise_label[repl_idx])
        noisy_rep_buffer["Images"].extend(warmup_dataset.train_data[repl_idx_noisy])

        # print(len(clean_buffer["Images"]))

        ## Fine-Tuning Stage
        if len(clean_buffer["Images"]) >= 500:

            X_images = clean_rep_buffer["Images"] + clean_buffer["Images"]
            X_labels = clean_rep_buffer["Labels"] + clean_buffer["Labels"]
            U_images = noisy_rep_buffer["Images"] + noisy_buffer["Images"]

            ### Main Training
            from dataloader_finetune import * 
            CEloss   = nn.CrossEntropyLoss()

            loader = cifar_dataloader(args.dataset, task_mode=task_mode_list,  r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size, num_workers=4,\
                root_dir=args.data_path, log=stats_log, noise_file='')

            labeled_trainloader, unlabeled_trainloader = loader.run(0.5, X_images, X_labels, U_images, 'train') 
            optimizer1 = optim.SGD(net1.parameters(), lr=args.lr*50, momentum=0.9, weight_decay=5e-4) 
            scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, args.finetune_epochs, 2e-2)
            best_acc_finetune = 0
            for epoch in range(args.finetune_epochs):
                train(epoch, net1, optimizer1, labeled_trainloader, unlabeled_trainloader)    # train net1  
                test_loader = loader.run(0, X_images, X_labels, U_images, 'test')
                acc, loss   = test(epoch, net1)

                if acc > best_acc_finetune:
                    model_name_1 = 'Net1_final.pth'
                    print("Save the Model --- --")
                    checkpoint1 = {
                        'net': net1.state_dict(),
                        'Model_number': 1,
                        'Noise_Ratio': args.r,
                        'Loss Function': '3 type',
                        'Optimizer': 'SGD',
                        'Noise_mode': args.noise_mode,
                        'Accuracy': acc,
                        'Dataset': args.dataset,
                        'Batch Size': args.batch_size,
                        'epoch': epoch,
                    }
                    torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
                    best_acc_finetune = acc

            ### RE-INTIALIZE
            clean_buffer = {"Images": [],
                "Labels": []}
            noisy_buffer = {"Images": []}