# Import necessary libraries
from tree import BinNodePenaltyOptimalTreeClassifier
from tree import OptimalTreeClassifier
import pandas as pd
import numpy as np
from dataset import load_data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz

# Function to remove constant columns
def remove_constant_columns(df):
    return df.loc[:, (df != df.iloc[0]).any()]

# Load the dataset
data_path = "./data/banknote_train.csv"
car_data = load_data(data_path)

# Sample a subset of the data for initial inspection (optional)
car_data_sub = car_data.sample(n=20, random_state=1)

# Remove constant columns
#car_data = remove_constant_columns(car_data)

# Separate features and target from the dataset
X = car_data.iloc[:, :-1]
y = car_data.iloc[:, -1]

#X = X.fillna(X.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Define the classifier
opt_tree = BinNodePenaltyOptimalTreeClassifier(max_depth=3,
                                 min_samples_leaf=1,
                                 alpha=0.01,
                                 criterion="gini",
                                 solver="gurobi",
                                 time_limit=20,
                                 verbose=True,
                                 solver_options={'mip_cuts': 'auto',
                                                 'mip_gap_tol': 0.8,
                                                 'mip_focus': 'balance'})

# Fit the model
opt_tree.fit(X=X, y=y)

# Make predictions
y_pred = opt_tree.predict(X=X_test)
y_pred_prob = opt_tree.predict_proba(X=X_test)

# Check confusion matrix
print("Confusion Matrix :")
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
print(classification_report(y_true=y_test, y_pred=y_pred))

# Get feature names
feature_names = X_train.columns

# Get class names
class_names = np.unique(y).astype(str)

# Ensure n_features_in_ is set correctly
if not hasattr(opt_tree, 'n_features_in_'):
    opt_tree.n_features_in_ = X_train.shape[1]

# Generate DOT data for visualization
dot_data = tree.export_graphviz(opt_tree,
                                out_file=None,
                                feature_names=feature_names,
                                class_names=class_names,
                                label='all',
                                impurity=True,
                                node_ids=True,
                                filled=True,
                                rounded=True,
                                leaves_parallel=True,
                                special_characters=False)

# Render the tree
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render(filename='optimal_tree', directory="/home/lipn_uspn/Desktop/Stage/code/optimaltree", view=True)
