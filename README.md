# SIN393_Introduction-to-computer-vision_2023


## Creating a conda environment for the course
---

* Windows - without GPU.
```
    $ conda create -n env-sin393-py39 python=3.9
    $ conda activate env-sin393-py39

    $ pip install notebook
    $ pip install matplotlib
    $ pip install scikit-image
    $ pip install scikit-learn
    $ pip install pandas
    $ pip install seaborn
    $ pip install ipympl
```
```
    $ conda env export > env-sin393-py39.yml
    $ conda list --explicit > env-sin393-py39.txt

    $ conda env create -f env-sin393-py39.yml 
```
```

* Linux - with GPU.

```
    $ conda create -n env-sin393-gpu-py39 python=3.9
    $ conda activate env-sin393-py39

    $ conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    $ conda install chardet

    $ pip install scikit-image
    $ pip install matplotlib
    $ pip install scikit-learn
    $ pip install notebook

    $ pip install seaborn

```

