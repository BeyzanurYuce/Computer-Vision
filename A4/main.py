import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load data for classification
data_classification = pd.read_csv('C:\\Users\\beyza\\PycharmProjects\\vision_cr\\data\\heart.csv')  # Replace with your file path

# Assuming 'X_classification' contains features and 'y_classification' contains labels
X_classification = data_classification.drop(columns='target')  # Replace 'label_column' with your label column name
y_classification = data_classification['target']

# Define classifiers for classification task
logistic_classifier = LogisticRegression()
svm_classifier = SVC()
random_forest_classifier = RandomForestClassifier()

classifiers = [logistic_classifier, svm_classifier, random_forest_classifier]

# Perform five-fold cross-validation for each classifier
for clf in classifiers:
    scores = cross_val_score(clf, X_classification, y_classification, cv=5, scoring='accuracy')
    print(f"Accuracy for {clf.__class__.__name__}: {scores.mean()}")

# Optionally, train and test the best-performing classifier on separate training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)
# best_classifier.fit(X_train, y_train)
# accuracy = best_classifier.score(X_test, y_test)
# print(f"Accuracy on test data: {accuracy}")
