# Classification on thermal images

In order to run the code in this repository, download the dataset of thermal images at https://drive.google.com/drive/folders/1BkbsIdJeAdiAsXj1-unRo4MF8O3G1bpv?usp=drive_link

The dataset should be split into training and inference partitions. Use the training partition for the training task, the algorithm will take care of splitting it into training and testing datasets. Use the inference partition to simulate a real-world inferencing condition. Each partition must be organized into two sub-folders: `free` and `infested`. The following is an example of how to organize the folders:
```
thermal_classifier
├── dataset
│   ├── train
│   │   ├── free
│   │   │   ├── ...
│   │   │
│   │   └── infested
│   │   │   ├── ...
│   │
│   └── infer
│   │   ├── free
│   │   │   ├── ...
│   │   │
│   │   └── infested
│   │   │   ├── ...
```

## Examples

### Training
Train a model:
```
python main.py --train --training-dataset ./dataset/train --save-model model.pkl --save-scaler scaler.pkl
```
This command will save the trained [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) model and fitted [scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) at `model.pkl` and `scaler.pkl` respectively.

### Inference
Perform inference:
```
python main.py --infer --inference-dataset ./dataset/infer --model model.pkl --scaler scaler.pkl
```
This command will use the model `model.pkl` and scaler `scaler.pkl` to predict samples from the `inference` partition.
