# SIN-393 - Introduction to computer vision (2023)

### Prof. João Fernando Mari ([*joaofmari.github.io*](https://joaofmari.github.io/))
---

## Lecture 01 - Classifying images
---
* [Slides (EN)](/slides-en/Lecture01.ClassifyingImagens.(2023).pdf)
* [Slides (PT-BR)](/slides/Aula01.ClassificandoImagens.(2023).pdf)
* [Notebook (EN) - Part 1](/notebooks/Lecture%2004%20-%20Part%201%20-%20Convolutional%20Neural%20Networks.ipynb)
* [Notebook (EN) - Part 2](/notebooks/Lecture%2001%20-%20Part%202%20-%20Image%20classification.ipynb)
* [Notebook (EN) - Part 3](/notebooks/Lecture%2001%20-%20Part%203%20-%20Image%20classification%20-%20Color%20features.ipynb)

## Lecture 02 - Introduction to Python
---
* [Notebook (EN)](/notebooks/Lecture%2002%20-%20Introduction%20to%20Python.ipynb)

## Lecture 03 - Introdução ao NumPy
---
* [Notebook (EN)](/notebooks/Lecture%2003%20-%20Introduction%20to%20NumPy.ipynb)

## Lecture 04 - Artificial Neural Networks 
---
* [Slides (EN)](/slides-en/Lecture06.ConvolutionalNeuralNetworks.(2023).pdf)
* [Slides (PT-BR)](/slides/Aula06.RedesNeuraisConvolucionais.(2023).pdf)
* [Notebook (EN) - Part 1](/notebooks/Lecture%2004%20-%20Part%201%20-%20Convolutional%20Neural%20Networks.ipynb)
* [Notebook (EN) - Part 2](/notebooks/Lecture%2004%20-%20Part%202%20-%20Convolutional%20Neural%20Networks.ipynb)
* [Notebook (EN) - Part 3](/notebooks/Lecture%2004%20-%20Part%203%20-%20Convolutional%20Neural%20Networks.ipynb)
* [Notebook (EN) - Part 4](/notebooks/Lecture%2004%20-%20Part%204%20-%20Convolutional%20Neural%20Networks.ipynb)


## Lecture 05 - Artificial Neural Networks and Deep Learning
---
* In progress...

## Lecture 06 - Convolutional Neural Networks
---
* [Slides (EN)](/slides-en/Lecture06.ConvolutionalNeuralNetworks.(2023).pdf)
* [Slides (PT-BR)](/slides/Aula06.RedesNeuraisConvolucionais.(2023).pdf)
* [Notebook (EN) - Part 1](/notebooks/Lecture%2004%20-%20Part%201%20-%20Convolutional%20Neural%20Networks.ipynb)
* [Notebook (EN) - Part 2](/notebooks/Lecture%2004%20-%20Part%202%20-%20Convolutional%20Neural%20Networks.ipynb)
* [Notebook (EN) - Part 3](/notebooks/Lecture%2004%20-%20Part%203%20-%20Convolutional%20Neural%20Networks.ipynb)
* [Notebook (EN) - Part 4](/notebooks/Lecture%2004%20-%20Part%204%20-%20Convolutional%20Neural%20Networks.ipynb)

# Creating a conda environment for the course
---

## Without GPU
```
    $ conda create -n env-sin393-cpu-py39 python=3.9
    $ conda activate env-sin393-cpu-py39

    $ conda install pytorch torchvision torchaudio cpuonly -c pytorch
    
    $ pip install notebook
    $ pip install matplotlib
    $ pip install scikit-image
    $ pip install scikit-learn
    $ pip install pandas
    $ pip install seaborn
    $ pip install ipympl
    $ pip install ipywidgets
    $ pip install graphviz
```

## With GPU
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
    $ pip install ipywidgets
```

## You can easily save and load a conda environment:

* Saving an environment:
```
    $ conda env export > env-sin393-gpu-py39.yml
```

* Loading an environment:
```
    $ conda env create -f env-sin393-gpu-py39.yml 
```

* I provide YML files for the CPU and GPU conda environment.
```
    env-sin393-cpu-py39.yml
    env-sin393-gpu-py39.yml
```

# Bibligraphy

* GONZALEZ, R.C.; WOODS, R.E. Digital Image Processing. 3. ed. Pearson, 2007.
* COSTA, L.F.; CÉSAR-JR., R.M. Shape analysis and classification: Theory and practice. 1. ed. CRC Press, 2000.
* DUDA, R.O.; HART, P.E.; STORK, D.G. Pattern Classification. Wiley, 2001. 
* CHITYALA, R.; PUDIPEDDI, S. Image processing and acquisition using Python. CRC Press, 2014.
* MARQUES FILHO, O.; VIEIRA NETO, H. Processamento digital de imagens. Brasport, 1999.
* PONTI, M; COSTA, G. B. P. Como funciona o Deep Learning. Computer Vision and Pattern Recognition. 2017.
    * https://arxiv.org/abs/1806.07908  
    * http://dainf.ct.utfpr.edu.br/~hvieir/pub.html   
* PONTI, M. Everything you wanted to know about Deep Learning for Computer Vision but were afraid to ask. SIBGRAPI 2017 Tutorial.
    * http://conteudo.icmc.usp.br/pessoas/moacir/p17sibgrapi-tutorial/  
* CS231n: Convolutional Neural Networks for Visual Recognition.
    * http://cs231n.stanford.edu/ 
* GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. Deep Learning. MIT Press, 2016. 
    * http://www.deeplearningbook.org 
* NIELSEN, M. Neural Networks and Deep Learning. On-Line Book. 
    * http://neuralnetworksanddeeplearning.com  



# How to cite

* How to cite this material:

```
    @misc{mari_comp_vis_2023,
        author = {João Fernando Mari},
        title = {Introduction to computer vision},
        year = {2023},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/joaofmari/SIN393_Introduction-to-computer-vision_2023}}
    }
```

---
João Fernando Mari - UFV-CRP - 2022-2 - [joaofmari.github.io](joaofmari.github.io) - joaof.mari@ufv.br
