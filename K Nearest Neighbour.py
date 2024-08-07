#################################################################################
# THIS IS JUST A BASIC IMPLLEMENTATION
# IF THERE ARE ANY CORRECTIONS PLEASE LET ME KNOW
#################################################################################

# KNN to predict weather a person has diabetes or not

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
# used to make mean = 0 and standard deviation = 1 => Normally distributed
# Many algorithms assume the data to be normally distributed
# To satisfy the condition the 'StandardScalar' module makes the data normal
# This changes the data but improves the accuracy of the answer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
dataset = pd.read_csv('diabetes.csv')
print(dataset)
print(len(dataset))
print(dataset.head())

# We cannot accept zeros as inputs in place of 'Glucose', 'BloodPressure'
# 'SkinThickness', 'BMI', 'Insulin' as it will affect the final result
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)
    print("mean", mean)
    # mean of individual datasets
print(dataset)
print(len(dataset))
print(dataset.head())

# now after cleaning we split the data into training and test data set

X = dataset.iloc[:, 0:8]  # all rows with columns 0 to 8
y = dataset.iloc[:8]  # all rows and column 8
# column 8  'Outcome' is the output

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Feature Scaling
sc_X = StandardScaler()  # basically getting z-values
# This line converts mean to 0 and standard deviation to 1
# This is done by subtracting the mean of the data from each data point and then dividing by the standard deviation.

# Mathematics behind StandardScalar
# The `StandardScaler` class from the scikit-learn library is used to standardize features by removing the mean and
# scaling to unit variance. This involves calculating the mean and standard deviation of each feature in the data, and
# then using these values to transform the data.

# The formula for standardizing a feature `x` is as follows:
# z = (x - mean) / standard_deviation

# where `mean` is the mean of the feature `x` and `standard_deviation` is its standard deviation.
# For example, let's say we have a feature `x` with the following values:
# [1, 2, 3, 4, 5].
# The mean of this feature is
# `(1 + 2 + 3 + 4 + 5) / 5 = 3`,
# and its standard deviation is
# `sqrt(((1 - 3)^2 + (2 - 3)^2 + (3 - 3)^2 + (4 - 3)^2 + (5 - 3)^2) / 5) =
# sqrt(2.5) = 1.58`.
# Using the formula above, we can standardize the feature `x` as follows:
# z = (x - mean) / standard_deviation
#   = ([1, 2, 3, 4, 5] - 3) / 1.58
#   = [-1.27, -0.63, 0, 0.63, 1.27]
# After standardization, the feature `x` has zero mean and unit variance.

X_train = sc_X.fit_transform(X_train)
# X_train = sc_X.fit_transform(X_train): This line of code performs two actions: `fit` and `transform`.
# The `fit` action calculates the mean and standard deviation of each feature (column) in the training data.
# The `transform` action then uses these values to standardize the training data.

# Standardizing the data means that for each feature (column), the mean is subtracted from each value and
# then the result is divided by the standard deviation. This process changes the distribution of the data
# so that it has a mean of 0 and a standard deviation of 1.

# The `fit_transform` method is a shortcut that performs both actions (`fit` and `transform`) in one step.
# After this line of code is executed, the variable `X_train` will contain the standardized training data.

# Example:
# Let's say we have the following training data for two features `A` and `B`:

# A   B
# 1   4
# 2   5
# 3   6

# First, we create an instance of the `StandardScaler` class:

# sc_X = StandardScaler()
# Next, we use the `fit_transform` method to standardize the training data:

# X_train = np.array([[1, 4], [2, 5], [3, 6]])
# X_train = sc_X.fit_transform(X_train)

# The `fit` action calculates the mean and standard deviation for each feature. For feature `A`, the mean is
# `(1 + 2 + 3) / 3 = 2` and the standard deviation is
# `sqrt(((1 - 2) ** 2 + (2 - 2) ** 2 + (3 - 2) ** 2) / 3) = 1`.
# For feature `B`,
# the mean is `(4 + 5 + 6) / 3 = 5`
# and the standard deviation is
# `sqrt(((4 - 5) ** 2 + (5 - 5) ** 2 + (6 - 5) ** 2) / 3) = 1`.

# The `transform` action then uses these values to standardize the training data.
# For feature `A`, the standardized values are `[(1 - 2) / 1, (2 - 2) / 1, (3 - 2) / 1] = [-1, 0, 1]`.
# For feature `B`, the standardized values are `[(4 - 5) / 1, (5 - 5) / 1, (6 - 5) / 1] = [-1,0,1]`.

# After the `fit_transform` method is executed, the variable `X_train` will contain the standardized training data:

# array([[-1., -1.],
#        [0.,   0.],
#        [1.,   1.]])

X_test = sc_X.transform(X_test)
# X_test = sc_X.transform(X_test) is a line of code that uses the transform method of the
# StandardScaler object sc_X to standardize the test data.

# The transform method uses the mean and standard deviation calculated from the training data
# (by the fit method) to standardize the test data. For each feature in the test data, the mean
# (calculated from the training data) is subtracted from each value and then the result is divided
# by the standard deviation (also calculated from the training data).

# After this line of code is executed, the variable X_test will contain the standardized test data.

# TODO: This Note is very important (1)
# It’s important to note that the test data is transformed using the mean and standard deviation
# calculated from the training data, not from the test data itself. This ensures that the scaling applied
# to the test data is consistent with the scaling applied to the training data.

# Define the model: KNN
classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidian')
# This line of code creates an instance of the KNeighborsClassifier class from the sklearn.neighbors module
# and assigns it to the variable classifier. The KNeighborsClassifier is a type of classifier that uses the
# k-nearest neighbors algorithm to make predictions.

# The n_neighbors parameter specifies the number of nearest neighbors to include in the majority voting
# process. In this case, n_neighbors is set to 11, so the classifier will consider the 11 nearest
# neighbors when making a prediction.

# The p parameter specifies the power parameter for the Minkowski metric. When p=2, this is equivalent
# to using the Euclidean distance metric. The distance metric is used to calculate the distance between
# data points when finding the nearest neighbors.

# The metric parameter specifies the distance metric to use for calculating distances between data points.
# In this case, metric is set to 'euclidian', but it should be noted that the correct spelling for this
# distance metric is 'euclidean'.

# After this line of code is executed, the variable classifier will contain an instance of the
# KNeighborsClassifier class that is ready to be trained on data and used to make predictions.

# Prediction
y_pred = classifier.predict(X_test)
# y_pred = classifier.predict(X_test) is a line of code that uses the predict method of the classifier
# object to make predictions on the test data.

# The predict method takes as input the test data (X_test) and returns an array of predicted class labels
# (y_pred) for each sample in the test data. The predictions are made based on the k-nearest neighbors algorithm,
# which considers the n_neighbors nearest neighbors of each sample in the test data and assigns the most common
# class label among these neighbors as the prediction for that sample.
#
# After this line of code is executed, the variable y_pred will contain an array of predicted class
# labels for the test data.

# Evaluation Model
cm = confusion_matrix(y_test, y_pred)
# The confusion_matrix function takes as input two arrays: the true class labels for the test data
# (y_test) and the predicted class labels for the test data (y_pred). The function returns a confusion
# matrix that shows the number of correct and incorrect predictions made by the classifier.

# The confusion matrix is a square matrix with dimensions equal to the number of classes in the data.
# Each row of the matrix represents the instances of an actual class, while each column represents the
# instances of a predicted class. The diagonal elements of the matrix represent the number of correct
# predictions (i.e., true positives and true negatives), while the off-diagonal elements represent the
# number of incorrect predictions (i.e., false positives and false negatives).

# After this line of code is executed, the variable cm will contain the confusion matrix for the classifier’s
# predictions on the test data.


# A confusion matrix is a table that is often used to evaluate the performance of a classifier.
# It shows the number of correct and incorrect predictions made by the classifier, broken down by class.
#
# The confusion matrix is a square matrix with dimensions equal to the number of classes in the data.
# Each row of the matrix represents the instances of an actual class, while each column represents the
# instances of a predicted class.
#
# The diagonal elements of the matrix represent the number of correct predictions
# (i.e., true positives and true negatives), while the off-diagonal elements represent the number of incorrect
# predictions (i.e., false positives and false negatives).
#
# For example, let’s say we have a binary classification problem with class labels 0 and 1. The confusion matrix
# for this issue would be a 2x2 matrix with the following structure:
#
# | TN | FP |
# | FN | TP |
# where TN is the number of true negatives (i.e., samples with actual class label 0 that were correctly
# predicted as 0), FP is the number of false positives (i.e., samples with actual class label 0 that were incorrectly
# predicted as 1), FN is the number of false negatives (i.e., samples with actual class label 1 that were incorrectly
# predicted as 0), and TP is the number of true positives (i.e., samples with actual class label 1 that were correctly
# predicted as 1).
#
# The confusion matrix provides a detailed breakdown of the classifier’s performance and can be used to calculate
# various performance metrics such as accuracy, precision, recall, and F1-score.
print(cm)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
