#!/usr/bin/env bash

kill $(ps aux | grep "main_lightsecagg.py" | grep -v grep | awk '{print $2}')

nohup sh run_lightsecagg_distributed_pytorch.sh 8 8 resnet56 homo 3 1 32 0.001 cifar10 ./../../../data/cifar10 sgd 0 > "./lightsecagg.$(date +%s).log" 2>&1 &