import numpy as np
from sklearn.utils import check_array

class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Creates an instance of CustomKMeans.
        :param n_clusters: Amount of target clusters (=k).
        :param max_iter: Maximum amount of iterations before the fitting stops.
        :param random_state: Initialization for randomizer.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        Main clustering method of the CustomKMeans class. The method performs the clustering on vectors given in X.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        # Input validation:
        X = check_array(X, accept_sparse='csr')

        # Calculation of cluster centers:
        self.cluster_centers_ = None

        #Initializes centroids randomly
        np.random.seed(self.random_state)
        initial_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = X[initial_indices]

        # Chooses number of n_clusters random points
        # Assigns random points as initial centroids

        for i in range(self.max_iter):
          # Assignment Step
          labels = []
          for x in X:
            distances = np.linalg.norm(x - self.cluster_centers_, axis=1)
            nearest_centroid_index = np.argmin(distances)
            labels.append(nearest_centroid_index)
          self.labels_ = np.array(labels)

        # Calculates the Euclidean distance between x and each of the cluster centers stored in self.cluster_centers_
        # Finds the index of the nearest cluster center for the data point x.
        # Appends the index of the nearest cluster center to the labels list.


         # Update Step
          new_centers = []
          for j in range(self.n_clusters):
            cluster_points = X[self.labels_ == j]
            new_center = cluster_points.mean(axis=0) if len(cluster_points) > 0 else self.cluster_centers_[j]
            new_centers.append(new_center)
          new_centers = np.array(new_centers)

          # Recalculate the cluster centers
          # This loop iterates over each cluster and calculates the new center as the mean of all points assigned to that cluster.
          # If a cluster has no points assigned, the previous center is retained. After recalculating the centers for all clusters,
          # the list of new centers is converted to a NumPy array and stored in 'new_centers'.

        # Convergence Check
          if np.all(np.equal(new_centers, self.cluster_centers_)):
            break
          self.cluster_centers_ = new_centers

        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels.
        """
        self.fit(X)
        return self.labels_

from sklearn.neighbors import NearestNeighbors

class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Creates an instance of CustomDBSCAN.
        :param min_samples: Equivalent to minPts. Minimum amount of neighbors of a core object.
        :param eps: Short for epsilon. Radius of considered circle around a possible core object.
        :param metric: Used metric for measuring distances.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        Main clustering method of the CustomDBSCAN class. The method performs the clustering on vectors given in X.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        # Input validation:
        X = check_array(X, accept_sparse='csr')

        # Determination of labels:
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        # Label all points
        neighbors_model = NearestNeighbors(radius=self.eps)
        neighbors_model.fit(X)
        n_neighbors = neighbors_model.radius_neighbors(X, return_distance=False) #for each point in X, its neighbors within the radius=0.5 are found.

        is_core = np.array([len(n) >= self.min_samples for n in n_neighbors])
        labels = -np.ones(X.shape[0], dtype=int)  # initially all labels are -1 (indicating that they are noise points)

        # Eliminate noises and connect core points
        cluster_id = 0
        for i in range(X.shape[0]):
            if not is_core[i] or labels[i] != -1:
                continue
            labels[i] = cluster_id
            neighbors = n_neighbors[i].tolist()
            for t in neighbors:
                if labels[t] == -1:
                    labels[t] = cluster_id
                    if is_core[t]:
                        neighbors.extend(n_neighbors[t].tolist())
            cluster_id += 1
        """
        * If the element of dataset X is not a core point or has already been assigned to a cluster, it skips to the next point.
        * Otherwise, it assigns the current cluster ID to the point and its neighbors.
        * For each neighbor, if it's a core point, its neighbors are also added to the list to be processed --> expands the cluster.
        * After processing all points in the cluster, the cluster_id is incremented for the next cluster.
        """

        self.labels_ = labels
        return self


    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_

import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = 'giraffe.png'
image = cv2.imread(image_path)

image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

height, width, _ = image.shape
pixel_data = image.reshape(-1, 3)

k = 4

kmeans = CustomKMeans(n_clusters=k)
kmeans_labels = kmeans.fit_predict(pixel_data)

cl_name = 'kmeans'
cl_labels = kmeans_labels
cluster_centers = {}
cluster_counts = {}

for i, label in enumerate(cl_labels):
    if label not in cluster_centers:
        cluster_centers[label] = [0, 0, 0]
        cluster_counts[label] = 0

    cluster_centers[label] = [sum(x) for x in zip(cluster_centers[label], pixel_data[i])]
    cluster_counts[label] += 1

for label in cluster_centers:
    cluster_centers[label] = [x / cluster_counts[label] for x in cluster_centers[label]]



# DBSCAN
# Setting hyperparameter(s):
eps = 5
min_pts = 30

# Performing the clustering:
dbscan = DBSCAN(eps=eps, min_samples=min_pts)
dbscan_labels = dbscan.fit_predict(pixel_data)

# Getting average cluster colors for each clustering:
# (In theory, we can also use the centroids from k-means as average colors.
# Here we ignore that for the sake of a unified method.)
for cl_name, cl_labels in (('kmeans', kmeans_labels), ('dbscan', dbscan_labels)):
    cluster_centers = {}
    cluster_counts = {}

    for i, label in enumerate(cl_labels):
        if label not in cluster_centers:
            cluster_centers[label] = [0, 0, 0]
            cluster_counts[label] = 0

        cluster_centers[label] = [sum(x) for x in zip(cluster_centers[label], pixel_data[i])]
        cluster_counts[label] += 1

    for label in cluster_centers:
        cluster_centers[label] = [x / cluster_counts[label] for x in cluster_centers[label]]

    # Creating a new image with the cluster centers as pixel values:
    new_image = np.array([cluster_centers[label] for label in cl_labels], dtype=np.uint8)
    new_image = new_image.reshape(height, width, 3)
    _ = cv2.imwrite(f'{cl_name}_output.jpg', new_image)


# Displaying the images:
plt.figure(figsize=(15, 5))

# Original image:
image1 = cv2.imread(image_path)
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title('original')

# k-means image:
image2 = cv2.imread('kmeans_output.jpg')
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title(f'k-means ({k=})')

# DBSCAN image:
image3 = cv2.imread('dbscan_output.jpg')
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
plt.title(f'DBSCAN ({min_pts=}, {eps=})')

plt.show()