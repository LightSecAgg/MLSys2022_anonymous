## LightSecAgg: a Lightweight and Versatile Design for Secure Aggregation in Federated Learning
(Under Review)

### Implementation
We implement our LightSecAgg algorithm by reusing part of the source code from FedML.ai.
To review the core algorithmic implementation, please check `fedml_api/distributed/lightsecagg`.

### Running Script
```
cd ./fedml_experiments/distributed/lightsecagg

# MNIST
sh run_lightsecagg_distributed_pytorch.sh 4 4 lr hetero 10 1 64 0.1 mnist "./../../../data/MNIST" sgd 0

# CIFAR-10
sh run_lightsecagg_distributed_pytorch.sh 8 8 resnet56 homo 100 20 64 0.001 cifar10 ./../../../data/cifar10 adam 0
nohup sh run_lightsecagg_distributed_pytorch.sh 8 8 resnet56 homo 100 20 64 0.001 cifar10 ./../../../data/cifar10 adam 0 > ./lightsecagg_lr001.log 2>&1 &

```
