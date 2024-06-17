# Bidirectional Decoupled Distillation For Heterogeneous Federated Learning
Code for paper - **[Bidirectional Decoupled Distillation For Heterogeneous Federated Learning]**. Our code is highly borrowed from [**Federated Learning on Non-IID Data with Local-drift Decoupling and Correction**](https://github.com/gaoliang13/FedDC.git) (CVPR 2022) by Liang Gao, Huazhu Fu, et al.


## Prerequisite
* Install the libraries listed in requirements.txt
    ```
    pip install -r requirements.txt
    ```

## Datasets preparation
**We give datasets for the benchmark, including CIFAR10, CIFAR100, MNIST dataset.**




For example, you can follow the following steps to run the experiments:

```python example_code_FMLDKD.py```

1. Run the following script to run experiments on the MNIST dataset for all above methods:
    ```
    python example_code_FMLDKD.py -data minist
    ```
    
2. Run the following script to run experiments on CIFAR10 for all above methods:
    ```
    python example_code_FMLDKD.py -data cifar10
    ```
    
3. Run the following script to run experiments on CIFAR100 for all above methods:
    ```
    python example_code_FMLDKD.py -data cifar100
    ```
    
4. To show the convergence plots, we use the tensorboardX package. As an example to show the results which stored in "./Folder/Runs/CIFAR100_100_23_iid_":
    ```
    tensorboard --logdir=./Folder/Runs/CIFAR10_100_23_iid
    ```
    
    
## Generate IID and Dirichlet distributions:
Modify the DatasetObject() function in the example code.
CIFAR-10 IID, 100 partitions, balanced data
```
data_obj = DatasetObject(dataset='CIFAR10', n_client=100, seed=17, rule='iid', unbalanced_sgm=0, data_path=data_path)
```
CIFAR-10 Dirichlet (0.3), 100 partitions, balanced data
```
data_obj = DatasetObject(dataset='CIFAR10', n_client=100, seed=47, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)
```


## FMLDKD 
The FedDC method is implemented in ```utils_methods_FMLDKD.py```. The baseline methods are stored in ```utils_methods.py```.

### Citation

```

```
