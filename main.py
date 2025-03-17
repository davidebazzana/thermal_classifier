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

    parser.add_argument('-d', '--dataset', help="Path to dataset. The dataset folder must contain two sub-folders: free and infested.")
    parser.add_argument('-f', '--features', help="Path to the extracted features. These features will be used for training/inference.")
    parser.add_argument('-m', '--model', help="Path to trained model.")
    parser.add_argument('--save-features', help="File name to save the features extracted.")
    parser.add_argument('--save-model', help="File name to save the trained model.")

    args = parser.parse_args()

    dataset = args.dataset
    if dataset is None:
        if args.features is None:
            raise RuntimeError("No dataset provided. Please provide a dataset (--dataset).")
    else:
        if not dataset.endswith("/"):
            dataset += "/"
    
    if args.features is None:
        feature_data = []
        labels = []
        loader = DataLoader(dataset)
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
        y = np.expand_dims(np.array(labels), axis=1)
        labelled_X = np.hstack((X, y))
        if args.save_features is not None:
            joblib.dump(labelled_X, args.save_features)
    else:
        print(f'Loading features from: {args.features}')
        labelled_X = joblib.load(args.features)

    X = labelled_X[:, :-1]
    y = labelled_X[:, -1]

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
    
    print(f'testing...')
    # Prediction & Accuracy
    y_pred = clf.predict(X_test)
    accuracy_i = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred, average=None))

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.6f} seconds")
    

if __name__ == "__main__":
    main()
