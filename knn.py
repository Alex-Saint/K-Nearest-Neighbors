# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style
from collections import Counter

#---------------K NEAREST NEIGHBOR---------------
# Function to find K Nearest Neighbors
def k_nearest_neighbors(data, test, k):
	# Array to keep track of all the distances
	distances = []
	# For each type of data in the passed dataset
	for group in data:
		# For each feature of the given type
		for features in data[group]:
			# Get euclidean distance
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(test))
			# Add distance and classification for distance to distances list
			distances.append([euclidean_distance, group])

	# Get the k shortest distances from the test point
	votes = [i[1] for i in sorted(distances)[:k]]
	# Get the mode from the votes list
	guess = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k
	# Return result
	return guess, confidence
#------------------------------------------------

#---------------DATA DISC -> MEMORY---------------
# Get MNIST training data
df = pd.read_csv("MNIST_training.csv")
train_data = df.astype(int).values.tolist()

# Get MNIST test data
df = pd.read_csv("MNIST_test.csv")
test_data = df.astype(int).values.tolist()
#-------------------------------------------------

#---------------PREPROCESS THE DATA---------------
# Put training set into dictionary
train_set = {new_list: [] for new_list in range(10)}
for i in train_data:
	train_set[i[0]].append(i[1:785])

# Put test set into dictionary
test_set = {new_list: [] for new_list in range(10)}
for i in test_data:
	test_set[i[0]].append(i[1:785])
#-------------------------------------------------

#---------------FIND THE BEST K VAL---------------
# Counters to keep track along the way
correct = total = accuracy = bestAccuracy = bestK = 0
# Lists to categorize accuracies of K
badK = []
avgK = []
goodK = []
# Threshold for bad/okay accuracy
BAD_ACC, OKAY_ACC = 80, 85
# Test different values of K
for i in range(1, 100):
	if i % 2 == 0:
		continue
	# For debugging
	print("testing ", i)
	# Get each group of numbers
	for group in test_set:
		# Test each sample in each group
		for data in test_set[group]:
			# Guess what class test item is in
			guess, confidence = k_nearest_neighbors(train_set, data, k = i)
			# Test if guess is correct
			if group == guess:
				correct += 1
			# Increment total
			total += 1
	# Compute accuracy for tested K
	accuracy = 100 * (correct/total)
	print("{}'s accuracy was {}%".format(i, accuracy))
	# Graph red if terrible value of K
	if accuracy < BAD_ACC:
		badK.append([i, accuracy])
	# Graph yellow if okayish value of K
	elif accuracy < OKAY_ACC:
		avgK.append([i, accuracy])
	# Graph green if good value of K
	else:
		goodK.append([i, accuracy])
	# Save best K value
	if accuracy > bestAccuracy:
		bestAccuracy = accuracy
		bestK = i
# Turn each array in a numpy array
badK = np.array(badK)
avgK = np.array(avgK)
goodK = np.array(goodK)
#-------------------------------------------------

#---------------SHOW CONFIDENCE IN K---------------
# Lists for later graphs
right = []
wrong = []
guesses = []
# Counts to calculate accuracy later
correct = total = 0
# Iterate through test data
print("Testing Test Dataset Against Training Dataset Using Best K")
for group in test_set:
	# Test each sample in each group
	for data in test_set[group]:
		# Guess what class test item is in
		guess, confidence = k_nearest_neighbors(train_set, data, k = bestK)
		# If guess is correct
		if(guess == group):
			correct += 1
			guesses.append([group, guess, confidence])
			right.append([total, confidence])
		# If guess is incorrect
		else:
			guesses.append([group, guess, confidence])
			wrong.append([total, confidence])
		total += 1
# Turn each array in a numpy array
right = np.array(right)
wrong = np.array(wrong)
guesses = np.array(guesses)

# Print best k, accuracy with that k
print("Best Value for K: ", bestK)
print("Best Accuracy: ", 100 * (correct / total), "%")
# Print actual vs guess and confidence in answer
for attempt in guesses:
	print("Actual: ", attempt[0], " Guess: ", attempt[1], " Confidence: ", attempt[2])
#--------------------------------------------------

#---------------VISUALIZE FINDINGS---------------
# Add lines to make graph more readable
matplotlib.style.use('fivethirtyeight')
# Multiple graphs in 1 figure
fig, (accPlt, confPlt) = plt.subplots(1, 2)
# ACCURACY PLOT
# Label axis
accPlt.set(xlabel = "Values of K", ylabel = "Accuracy (%)")
# Title plot
accPlt.set_title("Accuracy Per K")
# Plot data
if(len(badK) > 0):
	label = "Accuracy < " + str(BAD_ACC) + "%"
	accPlt.scatter(badK[:,0], badK[:,1], color = 'r', label = label)
if(len(avgK) > 0):
	label = "Accuracy < " + str(OKAY_ACC) + "%"
	accPlt.scatter(avgK[:,0], avgK[:,1], color = 'y', label = label)
if(len(goodK) > 0):
	label = "Accuracy > " + str(OKAY_ACC) + "%"
	accPlt.scatter(goodK[:,0], goodK[:,1], color = 'g', label = label)
# Show labels in the legend
accPlt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left', borderaxespad = 0.0)

# CONFIDENCE PLOT
# Label axis
confPlt.set(xlabel = "Index Of Test Value", ylabel = "Confidence (%)")
# Title plot
label = "Confidence Per Tested Value (Best K: " + str(bestK) + ")"
confPlt.set_title(label)
# Plot data
if(len(right) > 0):
	confPlt.scatter(right[:,0], right[:,1], color = 'g', label = "Correct")
if(len(wrong) > 0):
	confPlt.scatter(wrong[:,0], wrong[:,1], color = 'r', label = "Incorrect")
# Show labels in the legend
confPlt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left', borderaxespad = 0.0)

# Show plots
plt.show()
#------------------------------------------------