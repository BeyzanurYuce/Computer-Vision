import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

data = pd.read_csv('C:\\Users\\beyza\\PycharmProjects\\vision_cr\\data\\cars.csv')

# handle missing values
data['MPG'] = data['MPG'].fillna(data['MPG'].mean())

# split features and target variable
X = data.drop(columns=['MPG'])
y = data['MPG']

numeric_col = ['Acceleration', 'Cylinders', 'Displacement', 'Horsepower', 'Model_Year', 'Weight']

# handle missing values
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_col),
    ])

models = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree', DecisionTreeRegressor()),
    ('SVR', SVR(max_iter=1000))
]

# calculate the accuracy
results = {}
for name, model in models:
    pipeline = make_pipeline(preprocessor, model)
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    average_accuracy = scores.mean()
    results[name] = average_accuracy

for model, acc in results.items():
    print(f'{model}: Average Accuracy = {acc:.6f}')