import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('creditcard.csv')

print(df.head())
print(df.info())
print(df.isnull().sum())

print("\nClass distribution:")
print(df['Class'].value_counts())

legit = df[df.Class == 0]
fraud = df[df.Class == 1]
print("\nLegit shape:", legit.shape)
print("Fraud shape:", fraud.shape)

print("\nAmount stats for legit transactions:")
print(legit.Amount.describe())
print("\nAmount stats for fraudulent transactions:")
print(fraud.Amount.describe())

print("\nMean values by class:")
print(df.groupby('Class').mean())

legit_sample = legit.sample(n=492, random_state=2)
df1 = pd.concat([legit_sample, fraud], axis=0)
print("\nSampled data head:")
print(df1.head())

print("Class distribution after sampling:")
print(df1['Class'].value_counts())
print("Mean values after sampling:")
print(df1.groupby('Class').mean())

X = df1.drop(columns='Class', axis=1)
Y = df1['Class']
print("Features shape:", X.shape)
print("Labels shape:", Y.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, stratify=Y, random_state=2
)
print("Train/Test shapes:", X_train.shape, X_test.shape)

model = SVC(kernel='rbf', C=1.0, random_state=2)
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data:', training_data_accuracy)

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on testing data:', testing_data_accuracy)


with open('fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")
