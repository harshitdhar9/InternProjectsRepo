import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

iris_data=pd.read_csv('Datasets/Iris.csv')

iris_data.head()
iris_data.shape
iris_data.dtypes
iris_data.info()

iris_data.drop(columns='Id')
iris.describe().T

iris.isnull().sum()

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
iris_Species=label.fit_transform(iris_Species)

corr=iris_data.corr()
sns.heatmap(corr,cmap='coolwarm',annot=True)
plt.show()

hue_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
iris_data['Species'] = iris_data['Species'].map(hue_mapping)
sns.pairplot(iris, hue='Species', diag_kind='kde')
plt.show()

#Model Training
X=iris_data.drop(columns=['Species'])
y=iris_data['Species']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression
Log=LogisticRegression()
Log.fit(X_train,y_train)

Lr_pred=Log.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, f1_score,accuracy_score

cm = confusion_matrix(y_test, Lr_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion matrix of Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Classification Report is:\n", classification_report(y_test, Lr_pred))
print("Accuracy Score:", accuracy_score(y_test, Lr_pred))

#KFold Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=1)
# Perform cross-validation on the model
scores = cross_val_score(Log, X, y, cv=kf)
plt.figure(figsize=(10, 6))
plt.boxplot(scores)
plt.xlabel("Fold")
plt.ylabel("Score")
plt.title("Cross-Validation Scores")
plt.show()
mean_score = scores.mean()

