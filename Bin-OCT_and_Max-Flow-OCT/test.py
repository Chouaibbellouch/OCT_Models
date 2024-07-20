import dataset
import mfoct as MaxFlow
import oct as OCT
import binoct as BIN
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz
from collections import defaultdict
import numpy as np

#democrat=0
#republican=1

timelimit = 3000
seed = 42
d = 3

train_ratio = 0.5
val_ratio = 0.25
test_ratio = 0.25

x, y = dataset.loadData('banknote_train')

# Since the banknote dataset contains continuous features, no need for one-hot encoding
# x_enc = dataset.oneHot(x)  # Commented out

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=seed)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                test_size=test_ratio/(test_ratio+val_ratio), random_state=seed)
# x_train_enc, x_test_enc, y_train, y_test = train_test_split(x_enc, y, test_size=1-train_ratio, random_state=seed)
# x_val_enc, x_test_enc, y_val, y_test = train_test_split(x_test_enc, y_test,
#                                                         test_size=test_ratio/(test_ratio+val_ratio), random_state=seed)

oct = BIN.binOptimalDecisionTreeClassifier(max_depth=d, min_samples_split=0, timelimit=timelimit)
oct.fit(x_train, y_train)

# Define feature and class names
feature_names = [f'Feature {i}' for i in range(x.shape[1])]
class_names = [str(i) for i in np.unique(y)]  # Modified to use unique labels from y

# Export the tree to DOT format
dot_data = oct.plot_tree(feature_names=feature_names, class_names=class_names)

# Render the tree
graph = graphviz.Source(dot_data.source)
graph.format = 'png'
graph.render('optimal_tree')

# Optionally display the tree
# graph.view()
