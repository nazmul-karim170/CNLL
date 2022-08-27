import os
import torch
from tensorboardX import SummaryWriter
from data import DataScheduler
import numpy as np

def remap_targets( targets, lab):
    remapped_targets = torch.zeros(targets.size(), dtype= torch.long)
    for t in range(targets.size()[0]):
        remapped_targets[t] = (targets[t] == targets[1])

    # remapped_targets = (targets == lab[1])

    return remapped_targets

def get_match_index(targets, class_idx):
    target_indices = []
    # new_targets = []

    for i in range(len(targets)):
        if targets[i] in class_idx:
            target_indices.append(i)
            # new_targets.append(np.where(np.array(class_idx) == targets[i])[0].item())

    return target_indices


def train_model(config, dataset, savepath, scheduler: DataScheduler, writer: SummaryWriter):
    saved_model_path = os.path.join(config['log_dir'], 'ckpts', str(config['corruption_percent']))
    os.makedirs(saved_model_path, exist_ok=True)

    num_classes =  2
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    if dataset in [ 'cifar100']:
        if config['superclass_noise']:
            asym_val = 'sup'
        else:
            asym_val = 'rand'
    else:
        if config['asymmetric_noise']:
            asym_val = 'asym'
        else:
            asym_val = 'sym'

    model = []
    
    with torch.no_grad():
        print("Processing Images .. ..", flush=True)
        image_features = []
        labels   = []
        noise_label = []
        prev_t = 0
        total = 0
        lab_uni_acc = []
        for step, ((x, y, corrupt, idx), t) in enumerate(scheduler):
            if t != prev_t:

                np.save(os.path.join(savepath,  str(dataset) + '_Train_images_'+str(task_name)+ '_' +  str(asym_val) + '_' + str(config['corruption_percent'])), np.array(image_features))
                np.save(os.path.join(savepath,  str(dataset) + '_Train_labels_'+str(task_name)+ '_' + str(asym_val) + '_' + str(config['corruption_percent'])), labels)
                np.save(os.path.join(savepath,  str(dataset) + '_Noise_labels_'+str(task_name)+ '_' + str(asym_val) + '_' + str(config['corruption_percent'])), noise_label)
                
                # print(torch.unique(torch.Tensor(lab_uni_acc)))
                scheduler.eval(dataset,savepath, t-1, torch.unique(torch.Tensor(lab_uni_acc)), model, writer, step + 1, eval_title='eval')
                total = 0
                image_features = []
                labels = []
                noise_label = []
                lab_uni_acc = []
                print("Train Dataset Saving Done .....  ....")

            images, lab, cor = x, y, corrupt
            lab_uni = torch.unique(lab)
            lab_uni_acc.extend(lab_uni)

            ## Making the Labels Binary
            # lab = remap_targets(lab, lab_uni)

            lab.type(torch.LongTensor)
            task_name ='task_'+ str(t)
            batch_size = images.size()[0]
            # print(torch.unique(lab))
            # print(batch_size)
            image_features.extend(list(images.data.cpu().numpy()))            # labels_t   = Variable(lab.cuda())
            labels.extend(lab.data.cpu())
            noise_label.extend(cor.data.cpu())
            total += batch_size
            prev_t = t


        np.save(os.path.join(savepath,  str(dataset) + '_Train_images_'+str(task_name)+ '_' +  str(asym_val) + '_' + str(config['corruption_percent'])), np.array(image_features))
        np.save(os.path.join(savepath,  str(dataset) + '_Train_labels_'+str(task_name)+ '_' + str(asym_val) + '_' + str(config['corruption_percent'])), labels)
        np.save(os.path.join(savepath,  str(dataset) + '_Noise_labels_'+str(task_name)+ '_' + str(asym_val) + '_' + str(config['corruption_percent'])), noise_label)
        scheduler.eval(dataset, savepath,t,torch.unique(torch.Tensor(lab_uni_acc)), model, writer, step + 1, eval_title='eval')
        print("Train Dataset Saving Done .....  ....")



