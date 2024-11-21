import pickle
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

data_dict = pickle.load(open('.\\first_part\\data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
max_length = max(len(lst) for lst in data_dict['data'])
data = [lst + [0] * (max_length - len(lst)) for lst in data_dict['data']]

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

rf = model.fit(x_train, y_train)

y_predict = model.predict(x_test)
from sklearn import tree
estimator = model.estimators_[1]
selected_tree = rf.estimators_[0]

# fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (10,2), dpi=900)
# for index in range(0, 1):
#     tree.plot_tree(rf.estimators_[index],
#                 #    feature_names = fn, 
#                 #    class_names=cn,
#                    filled = True,
#                    ax = axes[index])
#     axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
# fig.savefig('rf_5trees.png')
fig=plt.figure(figsize=(50,50))

_ = tree.plot_tree(rf.estimators_[0],filled=True,fontsize=10)

fig.savefig('1.png')
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
