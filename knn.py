import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random

style.use('fivethirtyeight')

dataset = {'k': [[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]]}

new_features = [5,7]



def k_nearest_neighbors(data, predict, k = 3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups')
	distances = []
	# For each type of data in the passed dataset
	for group in data:
		# For each feature of the given type
		for features in data[group]:
			# Get euclidean distance
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
			# Add distance and classification for distance to distances list
			distances.append([euclidean_distance, group])

	# Get the k shortest distances from the test point
	votes = [i[1] for i in sorted(distances)[:k]]
	# Get the mode from the votes list
	vote_result = Counter(votes).most_common(1)[0][0]
	# Return result
	return vote_result

df = pd.read_csv("MNIST_training.csv")
train_data = df.astype(int).values.tolist()

df = pd.read_csv("MNIST_test.csv")
test_data = df.astype(int).values.tolist()


train_set = {new_list: [] for new_list in range(10)}
test_set = {new_list: [] for new_list in range(10)}
# result = k_nearest_neighbors(dataset, new_features, k = 3)

for i in train_data:
	train_set[i[0]].append(i[1:785])

for i in test_data:
	test_set[i[0]].append(i[1:785])

correct = 0
total = 0

for group in test_set:
	for data in test_set[group]:
		vote = k_nearest_neighbors(train_set, data, k = 15)
		if group == vote:
			correct += 1
		total += 1

print("Accuracy: ", correct/total)

# for i in dataset:
# 	for ii in dataset[i]:
# 		plt.scatter(ii[0], ii[1], s = 100, color = i)
# plt.scatter(new_features[0], new_features[1], color = result)
# plt.show()