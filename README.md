<h2 align="center"> <a href="https://github.com/nazmul-karim170/CNLL-Continual_Learning_Noisy_Labels">CNLL: A Semi-Supervised Approach for Continual Learning with Noisy Labels </a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2312.09313-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2204.09881.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/nazmul-karim170/CNLL-Continual_Learning_Noisy_Labels/blob/main/LICENSE) 


</h5>

## [Paper](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Karim_CNLL_A_Semi-Supervised_Approach_for_Continual_Noisy_Label_Learning_CVPRW_2022_paper.pdf) 


## Code for Training 

### ![](resources/algorithm.png)

### System Dependencies
- Python >= 3.6.1
- CUDA >= 9.0 supported GPU

### Installation
Using virtual env is recommended.
```
$ conda create --name CNLL python=3.6
```
Install pytorch==1.7.0 and torchvision==0.8.1.
Then, install the rest of the requirements.
```
$ pip install -r requirements.txt
```

### First, generate the different tasks out of a single dataset
Users can perform task/class incremental learning in this manner. We create class-wise tasks where each task has M number of classes to deal with. Specify parameters in `config` yaml, `episodes` yaml files. Here config contains dataset description and episodes contain task information.

	python main.py --log-dir [log directory path] --c [config file path] --e [episode file path] --override "|" --random_seed [seed]

Run CIFAR10 asymmetric noise rate of 40% experiment-

	python main.py --log-dir ./data --c configs/cifar10_spr.yaml --e episodes/cifar10-split_epc1_asym_a.yaml --override "asymmetric_noise=True|corruption_percent=0.4";

Run CIFAR100 superclass symmetric noise rate of 40%  experiment. Noise labels can be generated within 20 superclasses or randomly.

	python main.py --log-dir ./data --c configs/cifar100_spr.yaml --e episodes/cifar100sup-split_epc1_a.yaml --override "superclass_noise=True|corruption_percent=0.4";



### Run CNLL Algorithm for Continual Noisy Label Learning on These Tasks

Make sure the ".npy" files for different tasks are in the same data folder. Check "data_path" argument in "Train_cifar_CNLL.py". Also, please make sure noise mode and noise ratio are consistent with the task specification. 

For the CIFAR10 asymmetric noise rate of 40% experiment-

	python Train_cifar_CNLL.py --dataset cifar10 --noise_mode asym --r 0.4
	
	
For CIFAR100 symmetric and superclass noise rate of 40% experiment-

	python Train_cifar_CNLL.py --dataset cifar100 --noise_mode sup --r 0.4	
	 
For the CIFAR100 symmetric and random noise rate of 40% experiment-

	python Train_cifar_CNLL.py --dataset cifar100 --noise_mode rand --r 0.4
	
Thanks! If you have any queries please send an email nazmul.karim170@gmail.com. If you find the implementation useful, please cite our paper!

    @InProceedings{Karim_2022_CVPR,
        author    = {Karim, Nazmul and Khalid, Umar and Esmaeili, Ashkan and Rahnavard, Nazanin},
        title     = {CNLL: A Semi-Supervised Approach for Continual Noisy Label Learning},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        month     = {June},
        year      = {2022},
        pages     = {3878-3888}
    }


 
