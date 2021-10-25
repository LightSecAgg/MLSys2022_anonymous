## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments
```

nohup sh run_lightsecagg_distributed_pytorch.sh 8 8 resnet56 homo 100 20 64 0.001 cifar10 ./../../../data/cifar10 adam 0 > ./lightsecagg_lr001.log 2>&1 &

nohup sh run_lightsecagg_distributed_pytorch.sh 8 8 resnet56 homo 100 20 64 0.01 cifar10 ./../../../data/cifar10 adam 0 > ./lightsecagg_lr01.log 2>&1 &

nohup sh run_lightsecagg_distributed_pytorch.sh 8 8 resnet56 homo 100 20 64 0.1 cifar10 ./../../../data/cifar10 adam 0 > ./lightsecagg_lr1.log 2>&1 &
```