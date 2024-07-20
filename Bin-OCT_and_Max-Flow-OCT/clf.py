import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
import numpy as np

# Load the dataset
file_path = '/data/dataset/dataset.data'
df = pd.read_csv(file_path, header=None, delimiter=',')

# Rename columns
df.columns = ['feature', 'class']

# Separate features and labels
X = df[['feature']].values
y = df['class'].values

# Fit a DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3, random_state=1)
clf.fit(X, y)

# Plot the tree
plt.figure(figsize=(10, 6))
plot_tree(clf, filled=True, rounded=True, feature_names=['Feature'], class_names=['0', '1'])
plt.show()
