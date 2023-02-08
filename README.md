# Seeing Through Obstructions with Diffractive Cloaking
### [Project Page](https://light.princeton.edu/publication/seeing-through-obstructions/) | [Paper](https://dl.acm.org/doi/abs/10.1145/3528223.3530185)

[Zheng Shi](https://zheng-shi.github.io/), [Yuval Bahat](https://sites.google.com/view/yuval-bahat/home), [Seung-Hwan Baek](https://www.shbaek.com/), [Qiang Fu](https://cemse.kaust.edu.sa/vcc/people/person/qiang-fu), [Hadi Amata ](https://cemse.kaust.edu.sa/people/person/hadi-amata), [Xiao Li](), [Praneeth Chakravarthula](https://www.cs.unc.edu/~cpk/), [Wolfgang Heidrich](https://vccimaging.org/People/heidriw/), [Felix Heide](https://www.cs.princeton.edu/~fheide/)

If you find our work useful in your research, please cite:
```
@article{Shi2022SeeThroughObstructions,
author = {Shi, Zheng and Bahat, Yuval and Baek, Seung-Hwan and Fu, Qiang and Amata, Hadi and Li, Xiao and Chakravarthula, Praneeth and Heidrich, Wolfgang and Heide, Felix},
title = {Seeing through Obstructions with Diffractive Cloaking},
year = {2022},
issue_date = {July 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {41},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3528223.3530185},
doi = {10.1145/3528223.3530185}}
```

## Requirements
This code is developed using Pytorch on Linux machine. Full frozen environment can be found in 'env.yml', note some of these libraries are not necessary to run this code. Other than the packages installed in the environment, our image formation model uses package [pado](https://github.com/shwbaek/pado) to simulate wave optics.   

## Data
In the paper we use [Places365](http://places2.csail.mit.edu/index.html) and [Cityscapes](https://www.cityscapes-dataset.com/) as the obstruction-free background scene. And they can be easily swtich to any other datasets of your choice. See 'train.py' for more details on the data augmentation we applied. For more details on depth-aware obstruction simulation, please refer to 'models/'. 

## Pre-trained Models and Optimized DOE Designs
Optimzed DOE Designs and pre-trained models are available under 'ckpts/' folder. Please refer to the supplemental documents for fabrication details.

## Sensor Capture Simulation and Reconstruction
We include a sample script that demonstrates our entire image formation and reconstruction process. You can run the 'inference.ipynb' notebook in Jupyter Notebook. The notebook will load the checkpoint and run the entire process. The simulated depth-dependent PSFs, simulated sensor capture, as well as reconstructed image will be displayed within the notebook.

## Training
We include 'train.sh' for training purpose. Please refer to 'config/' for optics and sensor specs. 

## License
Our code is licensed under BSL-1. By downloading the software, you agree to the terms of this License. 

## Questions
If there is anything unclear, please feel free to reach out to me at zhengshi[at]princeton[dot]edu.
