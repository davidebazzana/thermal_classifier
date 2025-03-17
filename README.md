# Classification on thermal images

In order to run the code in this repository, download the dataset of thermal images at https://drive.google.com/drive/folders/1BkbsIdJeAdiAsXj1-unRo4MF8O3G1bpv?usp=drive_link

The dataset should be organized as follows:
```
thermal_classifier
├── dataset
│   ├── free
│   │   ├── ...
│   │
│   └── infested
│       ├── ...
```

## Examples

### Extract features and train
If no extracted features nor models are available, run the feature extraction and training:
```
python main.py --train --dataset ./dataset --save-features extracted_features.pkl --save-model model.pkl
```
This command will extract features using GLCM and save them in the file `extracted_features.pkl`. After that, it will train an SVC model and save it in the file `model.pkl`

### Train
You can train an SVC model on already extracted features by using the command:
```
python main.py --train --features extracted_features.pkl --save-model model.pkl
```
This command will skip the features extraction part, using the features available in the file `extracted_features.pkl`.
