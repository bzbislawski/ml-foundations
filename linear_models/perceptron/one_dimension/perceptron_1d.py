import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("percepstron_dataset_1d.csv")
data = dataset[['label', 'x', 'y']].values
data_x = dataset[['x']].values
data_y = dataset[['y']].values

t = 1.0
alpha = 0.1
for i in range(0, 100):
    for training_example in data:
        expected = training_example[0]
        x = training_example[1]

        predicted = x >= t if 1.0 else 0.0
        predicted = int(predicted)
        t = t + alpha * (float(predicted) - expected)
        t = round(t, 6)

print(t)

y = np.linspace(0, 10, 300)
x = 0 * y + t
plt.plot(x, y, '--r', label='t')

plt.plot(data_x, [0, 0, 0, 0], 'ro')
plt.axis([0, 10, 0, 10])
plt.title('Decision stump')
plt.xlabel('weight')
plt.ylabel('height')
plt.show()
