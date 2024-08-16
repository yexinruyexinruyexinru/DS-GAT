## Download datasets
+ DBLP
+ Tmall
+ Patent
  
All dataset can be found at [Dropbox](https://www.dropbox.com/sh/palzyh5box1uc1v/AACSLHB7PChT-ruN-rksZTCYa?dl=0). 
You can download the datasets and put them in the folder `data/`, e.g., `data/dblp`.

# Requirements

```
tqdm==4.59.0
scipy==1.10.1
texttable==1.6.2
torch==1.12.0+cu116
numpy==1.23.5
numba==0.56.4
scikit_learn==1.0
```
A specific package version is provided here, but other compatible versions can also reproduce the results of the paper. You may configure the setup according to your specific requirements.

## Build neighborhood sampler
```bash
python setup.py install
```
## Run DS-GAT

```bash
# DBLP
python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.4 --k 0.3
python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.6 --k 0.3
python main.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.8 --k 0.1

# Tmall
python main.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.4 --k 0.7
python main.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.6 --k 0.7
python main.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.8 --k 0.3

# Patent
python main.py --dataset patent --hids 512 10 --batch_size 2048 --p 1.0 --train_size 0.4 --k 0.9
python main.py --dataset patent --hids 512 10 --batch_size 2048 --p 1.0 --train_size 0.6 --k 0.5
python main.py --dataset patent --hids 512 10 --batch_size 2048 --p 0.5 --train_size 0.8 --k 0.7
```
## Ablation Study
The above commands provide the basic instructions for running the experiments. Additionally, we conducted several other experiments. To perform ablation studies, the following commands can be used. However, note that ablation studies are resource-intensive, especially on large datasets. Please ensure sufficient time and resources are available before proceeding.
```bash
# DBLP
python gat.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.4 
python gat.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.6 
python gat.py --dataset dblp --hids 128 10 --batch_size 1024 --p 0.5 --train_size 0.8 

# Tmall
python gat.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.4 
python gat.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.6 
python gat.py --dataset tmall --hids 512 10 --batch_size 1024 --p 1.0 --train_size 0.8 

# Patent
python gat.py --dataset patent --hids 512 10 --batch_size 2048 --p 1.0 --train_size 0.4 
python gat.py --dataset patent --hids 512 10 --batch_size 2048 --p 1.0 --train_size 0.6 
python gat.py --dataset patent --hids 512 10 --batch_size 2048 --p 0.5 --train_size 0.8 
```

