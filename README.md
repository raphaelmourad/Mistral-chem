# Mistral-DNA: Mistral large language model for chemical molecules

# Overview

Here is a repo to pretrain Mistral large language model for chemical molecules. Here the Mixtral model ([Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)) was modified to significantly reduce the number of parameters mostly by removing layers, such that it could be trained on a GPU such as an RTX3090.

# Requirements

If you have an Nvidia GPU, then you must install CUDA and cuDNN libraries. See:  
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html  
https://developer.nvidia.com/cudnn  
Be aware that you should check the compatibility between your graphic card and the versions of CUDA and cuDNN you want to install. 
This is a bit tricky and time consuming!

To know the version of your NVIDIA driver (if you use an NVIDIA GPU) and the CUDA version, you can type:  
```
nvidia-smi
```
The versions that were used here were : 
- NVIDIA-SMI 535.129.03
- Driver Version: 535.129.03
- CUDA Version: 12.2

The models were developed with python and transformers.  

Before installing python packages, you need to install python3 (>=3.10.12) (if you don't have it):  
```
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
```

Make mistral-dna environment:  
```
conda create -n mistral-chem python=3.8
conda activate mistral-chem
```

To install pytorch:  
```
pip install torch==2.2.1
```

Other python packages need to be installed:   
```
pip install transformers>=4.37.0.dev0 numpy>=1.24.4 pandas>=1.4.4 sklearn==0.0 datasets>=2.14.4 peft>=0.7.2.dev0
pip install flash-attn==0.2.4
pip install accelerate>=0.21.0
pip install bitsandbytes>=0.37.0
pip install progressbar
pip install tensorboard
pip install torch-xla==2.2.0
```

You might need to add this to your .bashrc:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your_path/mistral-chem/lib/
```

The pretrained model is available here:
```
https://huggingface.co/RaphaelMourad/Mistral-chem-v0.1
```


# Pretraining the model

Second, in the python folder "scriptPython/", you'll find the jupyter notebook:
- **script_pretrain_mistral_chem.ipynb** to pretrain Mixtral model on DNA sequences. \

Select the data you want to pretrain the model on 250k molecules.

The script can be ran on [Google Colab](https://colab.research.google.com/drive/1L2HaA5mopBr_77LNzU4-wJNjzCCPVGrL#scrollTo=W1AY86CaaAHd).

# Fine-tuning the model for classification

Third, in the python folder "scriptPython/", you'll find the jupyter notebook:
- **script_finetune_mistral_chem.ipynb** to finetune the pretrained Mixtral model on a specific classification task. \

To finetune the model, you must provide a dataset to train the model. 

The script can be ran on [Google Colab](https://colab.research.google.com/drive/1XJ7q1CLqmVUldzLIvEVshGU4oFHU_A_W#scrollTo=ID9BUyZ0qE_y).

# Contact: 
raphael.mourad@univ-tlse3.fr

