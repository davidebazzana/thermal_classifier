import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import *
from sklearn.preprocessing import *
from tqdm import tqdm
import os
import time
from matplotlib import pyplot as plt
import argparse
import joblib
from feature_extraction import *
from data_loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Thermal camera classification of bees varroa free/infested.")

    parser.add_argument('-t', '--train', action=argparse.BooleanOptionalAction, help="Train an SVC model")
    parser.add_argument('-i', '--infer', action=argparse.BooleanOptionalAction, help="Inference mode: classify images given scaler and model.")
    parser.add_argument('--training-dataset', help="Path to training dataset. The dataset folder must contain two sub-folders: free and infested.")
    parser.add_argument('--inference-dataset', help="Path to inference dataset. The dataset folder must contain two sub-folders: free and infested.")
    parser.add_argument('-m', '--model', help="Path to trained model.")
    parser.add_argument('-s', '--scaler', help="Path to scaler.")
    parser.add_argument('--save-model', help="File name to save the trained model to.")
    parser.add_argument('--save-scaler', help="File name to save the scaler to.")

    args = parser.parse_args()

    if args.train:
        training_dataset = args.training_dataset
        if training_dataset is None:
            raise RuntimeError("No training dataset provided. Please provide a dataset (--training-dataset).")
        else:
            if not training_dataset.endswith("/"):
                training_dataset += "/"
        
        """
        Feature extraction
        """
        feature_data = []
        labels = []
        loader = DataLoader(dataset_path=training_dataset)
        for image, label in tqdm(loader):
            mask = compute_image_entropy_mask(image)
            bounding_box = get_bounding_box(mask)
            if bounding_box is None:
                print(f'No bee detected')
            else:
                features = extract_features(image, bounding_box)
                feature_data.append(features)
                labels.append(label)
        X = [list(feat.values()) for feat in feature_data]
        X = np.array(X)
        X = X.reshape((X.shape[0], -1))
        y = np.expand_dims(np.array(labels), axis=1)
        
        labelled_X = np.hstack((X, y))
        X = labelled_X[:, :-1]
        y = labelled_X[:, -1]

        """
        Training
        """
        # Splitting dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        start_time = time.time()

        print(f'preprocessing...')
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        print(f'training...')
        # Train SVM Classifier
        clf = SVC(kernel='linear', class_weight='balanced')
        clf.fit(X_train, y_train)

        if args.save_model is not None:
            joblib.dump(clf, args.save_model)
            print("Model saved successfully.")

        if args.save_scaler is not None:
            joblib.dump(scaler, args.save_scaler)
            print("Scaler saved successfully.")
    
        print(f'testing...')
        # Prediction & Accuracy
        y_pred = clf.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("F1-score:", f1_score(y_test, y_pred, average=None))

        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.6f} seconds")

    if args.infer:
        inference_dataset = args.inference_dataset
        if inference_dataset is None:
            raise RuntimeError("No training dataset provided. Please provide a dataset (--training-dataset).")
        else:
            if not inference_dataset.endswith("/"):
                inference_dataset += "/"
        if not args.train:
            if args.model is None or args.scaler is None:
                raise RuntimeError("Provide model (--model) and scaler (--scaler) or train a model (--train).")
            else:
                scaler = joblib.load(args.scaler)
                clf = joblib.load(args.model)
        loader = DataLoader(dataset_path=inference_dataset)
        true_pred = []
        for image, label in tqdm(loader):
            mask = compute_image_entropy_mask(image)
            bounding_box = get_bounding_box(mask)
            if bounding_box is None:
                print(f'No bee detected')
            else:
                features = extract_features(image, bounding_box)
                x = list(features.values())
                x = np.array(x)
                x = np.ravel(x)
                x = np.expand_dims(x, axis=0)
                x = scaler.transform(x)
                y_pred = clf.predict(x)
                print(f'Prediction: {y_pred}')

if __name__ == "__main__":
    main()
