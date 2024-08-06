#################################################################################
# THEORY IS EXPLAINED FIRST THEN THE CODE IS GIVEN BELOW
#THIS IS JUST A BASIC IMPLLEMENTATION
# IF THERE ARE ANY CORRECTIONS PLEASE LET ME KNOW
#################################################################################


################################################
################################################
# THEORY
################################################
################################################

# USED FOR UNLABELED DATA SETS

# clustering helps identify two qualities of data:
# 1) Meaningfulness
# 2) Usefulness
# Meaningful clusters expand domain knowledge.
# For example, in the medical field, researchers applied clustering to gene expression experiments. The clustering
# results identified groups of patients who respond differently to medical treatments.
#
# Useful clusters, on the other hand, serve as an intermediate step in a data pipeline.
# For example, businesses use clustering for customer segmentation. The clustering results segment customers into
# groups with similar purchase histories, which businesses can then use to create targeted advertising campaigns.


# Clustering techniques:
# 1) Partition clustering
# 2) Hierarchical clustering
# 3) Density-based clustering

# 1) Partition Clustering

# Partition clustering divides data objects into non-overlapping groups. In other wrds, no object can be a member of
# more than one cluster, and every cluster must have at least one object.

# Partition clustering methods have several strengths:
# They work well when clusters have a spherical shape.
# They’re scalable with respect to algorithm complexity.

# They also have several weaknesses:
# They’re not well suited for clusters with complex shapes and different sizes.
# They break down when used with clusters of different densities.


# 2) Hierarchical Clustering

# It determines cluster assignments by building a hierarchy. This is implemented by either a bottom-up or a
# top-down approach:
# Agglomerate clustering is the bottom-up approach. It merges the two points that are the most similar until all
# points have been merged into a single cluster.

# Divisive clustering is the top-down approach. It starts with all points as one cluster and splits the least similar
# clusters at each step until only single data points remain.

# These methods produce a tree-based hierarchy of points called a dendrogram. Similar to partition clustering, in
# hierarchical clustering the number of clusters (k) is often predetermined by the user. Clusters are assigned by
# cutting the dendrogram at a specified depth that results in k groups of smaller dendrograms.

# The strengths of hierarchical clustering methods include the following:
# They often reveal the finer details about the relationships between data objects.
# They provide an interpretable dendrogram.

# The weaknesses of hierarchical clustering methods include the following:
# They’re computationally expensive with respect to algorithm complexity.
# They’re sensitive to noise and outliers.

# 3) Density-Based Clustering
# Density-based clustering determines cluster assignments based on the density of data points in a region.
# Clusters are assigned where there are high densities of data points separated by low-density regions.
# Unlike the other clustering categories, this approach does not require the user to specify the number of clusters.
# Instead, there is a distance-based parameter that acts as a tunable threshold. This threshold determines how close
# points must be to be considered a cluster member.

# The strengths of density-based clustering methods include the following:
# They excel at identifying clusters of non-spherical shapes.
# They’re resistant to outliers.

# The weaknesses of density-based clustering methods include the following:
# They aren’t well suited for clustering in high-dimensional spaces.
# They have trouble identifying clusters of varying densities.


# STEPS FOR K-MEAN CLUSTERING ALGORITHM
# Initialize the value of k which equals to the number of clusters you want to make
# Randomly initialize these points called cluster centroids in the graph
# It is an iterative algorithm, so it does two steps:
# 1) Pick a point and assign it to one of the centroids based on which is closed
# 2) Now move the centroid based on the average location of the points of that group
# This needs to be done until the centroids don't stop changing the positions

# The choosing of the initial points of centroids is done at random, everytime we run the algorithm on the same data
# the value changes.
# The quality of the cluster assignments is determined by computing the sum of the squared error (SSE) after the
# centroids converge, or match the previous iteration’s assignment. The SSE is defined as the sum of the squared
# Euclidean distances of each point to its closest centroid. Since this is a measure of error, the objective of
# k-means is to try to minimize this value.
# Researchers commonly run several initializations of the entire k-means algorithm and choose the cluster assignments
# from the initialization with the lowest SSE.

################################################
################################################
# CODE
################################################
################################################


import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Generating Synthetic data
features, true_labels = make_blobs(
    n_samples=200,                                       # n_samples is the total number of samples to generate.
    centers=3,                                           # centers is the number of centers to generate.
    cluster_std=2.75,                                    # cluster_std is the standard deviation.
    random_state=42
)


print(features[:5])
print(true_labels[:5])

# Normalizing the data
# Standardization scales, or shifts, the values for each numerical feature in your dataset so that the features have
# a mean of 0 and standard deviation of 1
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print(scaled_features[:5])

# Now the data are ready to be clustered.

kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)

# init controls the initialization technique. The standard version of the k-means algorithm is implemented by setting
# init to "random". Setting this to "k-means++" employs an advanced trick to speed up convergence, which you’ll use
# later.

# n_clusters sets k for the clustering step. This is the most important parameter for k-means.

# n_init sets the number of initializations to perform. This is important because two runs can converge on different
# cluster assignments. The default behavior for the scikit-learn algorithm is to perform ten k-means runs and return
# the results of the one with the lowest SSE.

# max_iter sets the number of maximum iterations for each initialization of the k-means algorithm.
kmeans.fit(scaled_features)


# Statistics from the initialization run with the lowest SSE are available as attributes of kmeans after calling .fit():

# The lowest SSE value
print(kmeans.inertia_)

# Final locations of the centroid
print(kmeans.cluster_centers_)

# The number of iterations required to converge
print(kmeans.n_iter_)

# Finally, the cluster assignments are stored as a one-dimensional NumPy array in kmeans.labels_. Here’s a look at the
# first five predicted labels:

print(kmeans.labels_[:5])


# CHOOSING APPROPRIATE VALUE OF K
# There are two methods:
# 1) elbow method
# 2) silhouette coefficient


# 1) elbow method
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

# There’s a sweet spot where the SSE curve starts to bend known as the elbow point. The x-value of this point is thought
# to be a reasonable trade-off between error and number of clusters. In this example, the elbow is located at x=3

plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

# Determining the elbow point in the SSE curve isn’t always straightforward. If you’re having trouble choosing the elbow
# point of the curve, then you could use a Python package, kneed, to identify the elbow point programmatically

kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
print(kl.elbow)


# 2) silhouette coefficient

# Silhouette coefficient values range between -1 and 1. Larger numbers indicate that samples are closer to their
# clusters than they are to other clusters.

#  The silhouette score() function needs a minimum of two clusters, or it will raise an exception.

# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)

# Plotting the average silhouette scores for each k shows that the best choice for k is 3 since it has the maximum score
plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()
