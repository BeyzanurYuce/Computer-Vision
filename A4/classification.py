import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv('C:\\Users\\beyza\\PycharmProjects\\vision_cr\\data\\heart.csv')

# Split features and target variable
X = data.drop(columns=['target'])
y = data['target']

numeric_col = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


# handle missing values
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_col)
    ])

models = [
    ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC()),
    ('KNN', KNeighborsClassifier())
]

# calculate the accuracy
results = {}
for name, model in models:
    pipeline = make_pipeline(preprocessor, model)
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    average_accuracy = scores.mean()
    results[name] = average_accuracy

for model, acc in results.items():
    print(f'{model}: Average Accuracy = {acc:.6f}')
