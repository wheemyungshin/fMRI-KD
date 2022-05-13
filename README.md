# fMRI-KD

This repository is python implementation code of fMRI-KD and depends on the repository : Generic Decoding Demo/Python (https://github.com/KamitaniLab/GenericObjectDecoding/tree/master/code/python)

## Requirements

All scripts are tested with Python 2.7.13.
The following packages are required.

- [bdpy](https://github.com/KamitaniLab/bdpy)
- numpy
- scipy
- pandas
- scikit-learn
- matplotlib (mandatory for creating figures)
- caffe (mandatory if you calculate image and category features by yourself)
- PIL (mandatory if you calculate image and category features by yourself)

## Data organization

All data should be placed in `data/` before training.
Data can be obrained from [figshare](https://figshare.com/articles/Generic_Object_Decoding/7387130).
The data directory should have the following files:

    data/ --+-- Subject1.h5 (fMRI data, subject 1)
            |
            +-- Subject2.h5 (fMRI data, subject 2)
            |
            +-- Subject3.h5 (fMRI data, subject 3)
            |
            +-- Subject4.h5 (fMRI data, subject 4)
            |
            +-- Subject5.h5 (fMRI data, subject 5)
            |
            +-- ImageFeatures.h5 (image features extracted with Matconvnet)

Download links:

- [Subject1.h5](https://ndownloader.figshare.com/files/15049646)
- [Subject2.h5](https://ndownloader.figshare.com/files/15049649)
- [Subject3.h5](https://ndownloader.figshare.com/files/15049652)
- [Subject4.h5](https://ndownloader.figshare.com/files/15049655)
- [Subject5.h5](https://ndownloader.figshare.com/files/15049658)
- [ImageFeatures.h5](https://ndownloader.figshare.com/files/15015971)

## Script files

- **train_teacher.py**: It trains 1-D resnet teacher using fMRI data. You should run this code first to generate teacher model. The save path is models/model1d_(subject_num)_weight_best.pt
- **train_both.py**: It trains 2-D resnet student using distillation from 1-D resnet teacher. Corresponding image dataset is required to run this code. (Warning!!)Due to copyright issues, the dataset is not open and only available by requesting at [https://forms.gle/ujvA34948Xg49jdn9](https://docs.google.com/forms/d/e/1FAIpQLSfuAF-tr4ZUBx2LvxavAjjEkqqUOj0VpqpeJNCe-IcdlqJekg/viewform) by the authors of [GenericObjectDecoding](https://github.com/KamitaniLab/GenericObjectDecoding).
- **train_both_feature_distill.py**: It trains 2-D resnet student using feature distillation from 1-D resnet teacher. Corresponding image dataset is required to run this code. (Warning!!)Due to copyright issues, the dataset is not open and only available by requesting at [https://forms.gle/ujvA34948Xg49jdn9](https://docs.google.com/forms/d/e/1FAIpQLSfuAF-tr4ZUBx2LvxavAjjEkqqUOj0VpqpeJNCe-IcdlqJekg/viewform) by the authors of [GenericObjectDecoding](https://github.com/KamitaniLab/GenericObjectDecoding).

For all train code, you can adjust config to select the target subjects, RoIs, etc.
