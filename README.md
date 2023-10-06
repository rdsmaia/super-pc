# super-pc
Introdução ao uso do supercomputador da UFRN
Neste repositório temos um simples conjunto de scripts para permitir o treinamento de um modelo de aprendizado profundo no supercomputador da UFRN (http://www.npad.ufrn.br/tutorials).

## Processo
1. ```ssh super-pc```
2. ```git clone https://github.com/rdsmaia/super-pc.git```
3. ```cd super-pc```
4. ```sbatch run_mnist_test.sh```
5. ```squeue -lu $USER```

## Ao final do treinamento
```nano slurm-{número_processo}.out```

