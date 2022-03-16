import numpy as np
from sklearn.cluster import KMeans

class DistributionEvaluation():
    """ Class for comparison of distributions using k-means.
    N_clusters cluster centers are fitted to the data_reference.
    For all datasets, the number of points in each cluster are computed and saved to
    self.histograms by calling self.process().
    Calling self.get_errors() computes the L2 distance to the reference histogram for every added dataset and saved to self.errors.
    Calling self.get_distances() computes the average distance of points in a cluster to the cluster center. This requires
       that self.process is called with do_distance_transform=True.
    """

    def __init__(self,
                 data_reference: np.array,  # data from distribution of reference
                 N_clusters: int  # number of clusters used in k-means
                 ):
        self.data = {}
        self.data["reference"] = data_reference
        self.N_clusters = N_clusters

    @property
    def N_clusters(self) -> int:
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
                do_distance_transform: bool = False  # if True: compute k-means transforms, required for distances
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

    def get_errors(self) -> dict:
        """ return distance between histograms of reference distribution and all other distributions. """
        errors = {key: compute_l2_distance(self.histograms["reference"], value)
                  for key, value in self.histograms.items()
                  if not key == "reference"}
        return errors

    def get_distances(self, *, std: bool = True, **kwargs) -> dict:
        """ get (combined) average (squared) distance to cluster centers (and standard deviation) for every cluster for all distributions.
            kwargs are identical to compute_distances, despite renorm
        """
        distance_ref = compute_distances(self.predictions["reference"], self.distance_transforms["reference"], self.N_clusters, std=False, **kwargs)
        _, std_ref = compute_distances(self.predictions["reference"], self.distance_transforms["reference"], self.N_clusters, std=True, renorm_dist=distance_ref, **kwargs)
        distances = {key: compute_distances(clusters, self.distance_transforms[key], self.N_clusters, renorm_dist=distance_ref, renorm_std=std_ref, std=std, **kwargs)
                     for key, clusters in self.predictions.items()
                     if not key == "reference"}
        return distances


def compute_distances(clusters: np.array, distances: np.array, N_clusters: int, *,
                      squared: bool = True,
                      std: bool = True,
                      combined: bool = False,
                      renorm_dist: float = 1.0,
                      renorm_std: float = 1.0) -> tuple:
    """ return for each cluster the average distance of data points to cluster center.

    Parameters
    ----------
    clusters : iterable
        contains clusters each data point belongs to.
    distances : numpy.array
        contains distances from each data point to each cluster center.
    N_clusters : int
        number of clusters
    squared : bool, default=True
        if False, compute mean
        if True, compute RMS
    std : bool, default=True
        if True, return standard deviation as well
    combined : bool, default=True
        if True, return average distance and standard deviation combined for all clusters
    renorm_dist : float or np.array(N_clusters)
        set to average_distance obtained for reference dataset
        resulting distances are renormalized by that
    renorm_std : float or np.array(N_clusters)
        set to standard_deviation obtained for (distance_renormed) reference dataset
        resulting standard_deviations are renormalized by that
    """
    distances = distances.min(1)  # distance to nearest cluster center
    if combined:
        average_distance = np.nanmean(distances**(1+squared))**(1-0.5*squared) # this requires nanmean, for clusters may be empty -> NaN distance
    else:
        average_distance = np.array([np.nanmean(distances[clusters == cluster]**(1+squared))**(1-0.5*squared) for cluster in range(N_clusters)])
    if not std:
        return average_distance / renorm_dist
    if combined:
        standard_deviation = np.nanmean(np.abs(distances - average_distance)**2)**0.5
    else:
        standard_deviation = np.array([np.nanmean(np.abs(distances[clusters == cluster] - avg_dist)**2)**0.5
                              for cluster, avg_dist in enumerate(average_distance)])
    return average_distance / renorm_dist, standard_deviation / renorm_std


def compute_l1_distance(reference, values, div=1) -> float:
    """ Estimate L2 distance between values and reference. """
    result = np.sum([np.abs(v - r)  / r for r, v in zip(reference, values)])
    return result / div


def compute_l2_distance(reference, values, div=1) -> float:
    """ Estimate L2 distance between values and reference. """
    result = np.sum([(v - r)**2 / r**2 for r, v in zip(reference, values)])
    return result / div