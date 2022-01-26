import numpy as np
from sklearn.cluster import KMeans


class DistributionEvaluation():
    """ Class for comparison of distributions using k-means.
    N_clusters cluster centers are fitted to the data_reference.
    For data_reference and other added datasets, the number of points in each cluster are computed and saved to
    self.histograms by calling self.process().
    Calling self.get_errors() computes the L2 distance to the reference histogram for every added dataset and saved to self.errors.
    Furthermore, the average distance of points in a cluster to the cluster center can be computed for every cluster by
     calling self.get_distances(). This requires that self.process is called with do_distance_transform=True.
    """

    def __init__(self,
                 data_reference: np.array,  # data from distribution of reference
                 N_clusters: int  # number of clusters used in k-means
                 ):
        self.data = {}
        self.data["reference"] = data_reference
        self.N_clusters = N_clusters

    @property
    def N_clusters(self):
        return self._N_clusters

    @N_clusters.setter
    def N_clusters(self, value):
        self._N_clusters = value
        # initialize k_means function accordingly
        self.k_means = KMeans(n_clusters=value, random_state=0).fit(self.data["reference"])

    def add(self,
            name: str,  # identifier for distribution
            data_points: np.array,  # data points from distribution
            ):
        """ add or replace dataset. """
        self.data[name] = data_points

    def process(self,
                do_distance_transform: bool = False  # if True: compute k-means transforms
                ):
        """ compute results after all distributions are given. """
        self.predictions, self.histograms, self.distance_transforms = {}, {}, {}
        for key, value in self.data.items():
            # compute nearest cluster center for every data point
            self.predictions[key] = self.k_means.predict(value)
            # count number of points in a cluster
            self.histograms[key] = np.bincount(self.predictions[key])
            if do_distance_transform:
                # compute distance from every data point to every cluster center
                self.distance_transforms[key] = self.k_means.transform(value)

    def get_errors(self):
        """ return distance between distograms of training distribution and all other distributions. """
        errors = {key: compute_l2_distance(self.histograms["reference"], value)
                  for key, value in self.histograms.items()
                  if not key == "reference"}
        return errors

    def get_distances(self, squared: bool = True):
        """ get average (squared) distance to cluster centers for all distributions. """
        distances = {key: compute_distances(clusters, self.distance_transforms[key], self.N_clusters, squared=squared)
                     for key, clusters in self.predictions.items()}
        return distances


def compute_distances(clusters: np.array, distances: np.array, N_clusters: int, squared: bool = True):
    """ return average distance to center for data points in every cluster.

    Parameters
    ----------
    clusters : iterable
        contains clusters each data point belongs to.
    distances : numpy.array
        contains distances from each data point to each cluster center.
    N_clusters : int
        number of clusters
    squared : bool, default=True
        if True, square distances
    """
    distances = distances.min(1)  # distance to nearest cluster center
    if squared:
        distances = distances * distances
    average_distances = [np.mean(distances[clusters == cluster]) for cluster in range(N_clusters)]
    return average_distances


def compute_l2_distance(reference, values, div=1):
    """ Estimate L2 distance between values and reference. """
    result = np.sum([(v - r) ** 2 / r for r, v in zip(reference, values)])
    return result / div