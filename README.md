
# Integrated Heterogeneous Graph Attention Network for Incomplete Multi-View Clustering  (IHGAT)

The code repository for "Integrated Heterogeneous Graph Attention Network for Incomplete Multi-View Clustering
" (the paper has been submitted to IEEE TIP) in PyTorch.

## Results
<img src='imgs/Table_High_Missing_Rate.png' width='300' height='130'>
<img src='imgs/Dimension_Analysis.png' width='300' height='240'>

Please refer to our [paper](Links will be provided when the paper is published) for detailed values.

## Prerequisites

The following packages are required to run the scripts:

Please see [INSTALL.md](./INSTALL.md)

## Dataset
We provide the source code on six benchmark datasets, i.e., CUB, Football, ORL, PIE, Politics and 3Sources. 

## Code Structures
There are four parts in the code.
 - `models`: It contains the backbone network and training protocols for the experiment.
 - `data`: Images and splits for the data sets.
- `dataloader`: Dataloader of different datasets.
 - `checkpoint`: The weights and logs of the experiment.
 
## Training scripts

- Train CIFAR100

  ```
  python train.py -projec fact -dataset cifar100  -base_mode "ft_cos" -new_mode "avg_cos" -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 600 -schedule Cosine -gpu 0,1,2,3 -temperature 16 -batch_size_base 256   -balance 0.001 -loss_iter 0 -alpha 0.5 >>CIFAR-FACT.txt
  ```
  
- Train CUB200
    ```
    python train.py -project fact -dataset cub200 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.25 -lr_base 0.005 -lr_new 0.1 -decay 0.0005 -epochs_base 400 -schedule Milestone -milestones 50 100 150 200 250 300 -gpu '3,2,1,0' -temperature 16 -dataroot YOURDATAROOT -batch_size_base 256 -balance 0.01 -loss_iter 0  >>CUB-FACT.txt 
    ```

- Train miniImageNet
    ```
    python train.py -project fact -dataset mini_imagenet -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 1000 -schedule Cosine  -gpu 1,2,3,0 -temperature 16 -dataroot YOURDATAROOT -alpha 0.5 -balance 0.01 -loss_iter 150 -eta 0.1 >>MINI-FACT.txt  
    ```

Remember to change `YOURDATAROOT` into your own data root, or you will encounter errors.

  

 
## Acknowledgment
We thank the following repos providing helpful components/functions in our work.

- [Awesome Few-Shot Class-Incremental Learning](https://github.com/zhoudw-zdw/Awesome-Few-Shot-Class-Incremental-Learning)

- [PyCIL: A Python Toolbox for Class-Incremental Learning](https://github.com/G-U-N/PyCIL)

- [Proser](https://github.com/zhoudw-zdw/CVPR21-Proser)

- [CEC](https://github.com/icoz69/CEC-CVPR2021)



## Contact 
If there are any questions, please feel free to contact with the author:  Da-Wei Zhou (zhoudw@lamda.nju.edu.cn). Enjoy the code.
