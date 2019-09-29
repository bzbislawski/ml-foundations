import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("linear_regression_dataset_2d.csv")
data = dataset[['label', 'x', 'y']].values
data_x = dataset[['x']].values
data_y = dataset[['y']].values

t = 1.0
w1 = w2 = 0
alpha = 0.1
output = []
for i in range(1, 100):
    for training_example in data:
        expected = training_example[0]
        x = training_example[1]
        y = training_example[2]

        predicted = w1 * x + w2 * y >= t if int(1.0) else int(0.0)
        predicted = int(predicted)
        w1 = w1 - alpha * (predicted - expected) * x
        w2 = w2 - alpha * (predicted - expected) * y
        t = t + alpha * (predicted - expected)
        w1 = round(w1, 6)
        w2 = round(w2, 6)
        t = round(t, 6)

print(w1, w2, t)
m = -(w1 / w2)
c = t / w2
print("f(x)= " + str(m) + "*x + " + str(c))

x = np.linspace(0, 10, 300)
y = m * x + c
plt.plot(x, y, '--r', label='t')

plt.plot(data_x, data_y, 'ro')
plt.axis([-2, 10, -2, 10])
plt.title('Decision stump')
plt.xlabel('weight')
plt.ylabel('height')
plt.show()
