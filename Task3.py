import pandas as pd
import numpy as np
import math

df = pd.read_csv("bank-full.csv", sep=';')

df = df[['age', 'job', 'marital', 'education', 'y']]  

for col in ['job', 'marital', 'education', 'y']:
    df[col] = df[col].astype('category').cat.codes

features = df.drop('y', axis=1).values
labels = df['y'].values

train_size = int(0.8 * len(df))
X_train, X_test = features[:train_size], features[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]

def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -sum(p * math.log2(p) for p in probs if p > 0)

def split_dataset(X, y, feature_index, value):
    left_X, right_X, left_y, right_y = [], [], [], []
    for i in range(len(X)):
        if X[i][feature_index] <= value:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])
    return np.array(left_X), np.array(right_X), np.array(left_y), np.array(right_y)

best_gain = 0
best_feature = 0
best_value = 0
base_entropy = entropy(y_train)

for feature_index in range(X_train.shape[1]):
    values = np.unique(X_train[:, feature_index])
    for val in values:
        left_X, right_X, left_y, right_y = split_dataset(X_train, y_train, feature_index, val)
        if len(left_y) == 0 or len(right_y) == 0:
            continue
        gain = base_entropy - (len(left_y)/len(y_train)) * entropy(left_y) - (len(right_y)/len(y_train)) * entropy(right_y)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature_index
            best_value = val

print(f" Best split is on feature index {best_feature} at value {best_value} with gain {best_gain:.4f}")

def predict(x):
    if x[best_feature] <= best_value:
        return 0  
    else:
        return 1  

correct = 0
for i in range(len(X_test)):
    pred = predict(X_test[i])
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)
print(f" Accuracy of 1-level decision tree: {accuracy * 100:.2f}%")

