import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    feature_file = 'feature.pkl'
    # semi_ml_pretrain.gen_features(feature_file)
    with open(feature_file, 'rb') as f:
        train_info = pickle.load(f)
    train_features = train_info['train_features']
    indices = train_info['indices']
    train_labels = train_info['train_labels']
    # Assuming `train_features` and `train_labels` are your features and labels respectively

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

    # Initialize the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the Decision Tree: {accuracy * 100:.2f}%")

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.show()


if __name__ == '__main__':
    main()
