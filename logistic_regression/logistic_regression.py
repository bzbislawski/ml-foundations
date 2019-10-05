import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score

dataset = pd.read_csv("dataset.csv")
x = dataset[['x', 'y']].values
y = dataset['label'].values

means_values = {}
scores = np.zeros([50,10])
# number of iterations when training
for i in range(50):
    # shuffling the data
    for iteration in range(10):
        for learning_rate in np.array([0.1, 0.01,0.001,0.0001]):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

            clf = SGDClassifier(loss='log', max_iter=i*2+1, alpha=learning_rate)
            clf.fit(x_train, y_train)
            scores[i][iteration] = clf.score(x_test, y_test)
            values = cross_val_score(clf,x,y,cv=5)
            print("Learning rate: " + str(learning_rate) + " values: " + str(np.mean(values)))
            means_values[str(learning_rate) + " " + str(i) + " " + str(iteration)] = np.mean(values)

print(scores)
plt.errorbar(list(np.arange(1,101,2)),np.mean(scores,axis=1),yerr=np.std(scores,axis=1))
plt.axis([0, 100, 0.6, 1.05])
plt.show()
print(means_values)

print(max(means_values, key=means_values.get))
