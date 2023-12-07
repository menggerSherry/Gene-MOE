# Gene-MOE

Gene-MOE: A Sparsely Gated Genomic Analysis Framework exploiting Pan-Cancer data.

## Gene-MOE Architecture

![avatar](figures/model.png)

## Install

### Requirements

* Python 3
* Pytorch 1.8.1 or higher
* Install numpy, tqdm, scikit-learn, imblearn, pandas and lifelines
```
pip install numpy
pip install tqdm
pip install scikit-learn
pip install pandas
conda install -c conda-forge imbalanced-learn
conda install -c conda-forge lifelines
```

## Run Weight Reduction

```
python train_encoder.py
```

## Run Survival Analysis
```
python train_cox.py
```


## Run Cancer Classification
```
python train_classifer.py
```

## dataset

[Google Drive](https://drive.google.com/file/d/1c83aQyPBWK-68i5gdDGFB0JtGi2XibOD/view?usp=sharing)
