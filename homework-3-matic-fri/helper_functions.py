import numpy as np


def UPGMA(distances):
    """Unweighted pair group method with arithmetic mean (UPGMA) agglomerative clustering.
    
    Parameters
    ----------
    distances: np.ndarray
        A two dimensional, square, symmetric matrix containing distances between data
        points. The diagonal is zeros.
        
    Returns
    -------
    np.ndarray
        The linkage matrix, as specified in scipy. Briefly, this should be a 2d matrix
        each row containing 4 elements. The first and second element should denote the
        cluster IDs being merged, the third element should be the distance, and the
        fourth element should be the number of elements within this new cluster. Any
        new cluster should be assigned an incrementing ID, e.g. after the first step
        where the first two points are merged into a cluster, the new cluster ID should
        be N, then N+1, N+2, ... in subsequent steps.
        
    Notes
    -----
    You can validate your implementation to scipy's `cluster.hierarchy.linkage`
    function using average linkage. UPGMA and average linkage agglomerative clustering
    are the same thing. Your function should return identical results in all cases.
    
    """
    
    if len(distances) != len(distances[0]):
        return -1
    
    result_dist = []
    clusters = []
    next_clusterId = len(distances)
    for i in range(len(distances)):
        clusters.append([i, 1])      # clusterId, number of members
    
    while len(distances) > 2:
        
        # find two closest nodes
        n1, n2, minDist = min_dist(distances)
        c1, c2 = clusters[n1], clusters[n2]
        #print(n1, n2)
        #print(clusters)

        new_cluster = (next_clusterId, c1[1]+c2[1])
        next_clusterId += 1
        new_distances = recalculate_distances(distances, n1, n2, c1, c2, new_cluster)

        # merge
        clusters, distances = remove_clustered(clusters, distances, n1, n2)
        clusters, distances = merge(clusters, distances, new_distances, new_cluster)
        
        result_dist.append([c1[0], c2[0], minDist, new_cluster[1]])
        
        #print(new_cluster)
        #print(distances)
        #print(clusters)

    # last cluster
    n1, n2, minDist = min_dist(distances)
    c1, c2 = clusters[n1], clusters[n2]
    new_cluster = (next_clusterId, c1[1] + c2[1])
    
    result_dist.append([c1[0], c2[0], minDist, new_cluster[1]])
    
    return result_dist


def jukes_cantor(p: float) -> float:
    """The Jukes-Cantor correction for estimating genetic distances.
    
    Parameters
    ----------
    p: float
        The proportional distance, i.e. the number of of mismatching symbols (Hamming
        distance) divided by the total sequence length.
        
    Returns
    -------
    float
        The corrected genetic distance.
    
    """
    return -0.75 * np.log(1 - 4/3 * p)

    
    
def min_dist(distances):
    minDist = np.finfo('d').max
    for i in range(len(distances)):
        for j in range(i+1, len(distances)):
            if distances[i][j] < minDist:
                minDist = distances[i][j]
                c1 = i
                c2 = j
    return c1, c2, minDist


def recalculate_distances(distances, node1, node2, cluster1, cluster2, new_cluster):
    new_distances = []
    for i in range(len(distances)):
        if i != node1 and i != node2:
            d1, d2 = distances[i][node1], distances[i][node2]
            d1_members, d2_members = cluster1[1], cluster2[1]
            new_distances.append((d1*d1_members + d2*d2_members)/(d1_members + d2_members))
            
    return new_distances


def remove_clustered(clusters, distances, node1, node2):
    del clusters[node1]
    del clusters[node2-1]
    distances = np.delete(np.delete(distances, [node1, node2], 0), [node1, node2], 1)
    return clusters, distances

    
def merge(clusters, distances, new_distances, new_cluster):
    clusters.append(new_cluster)    
    distances = np.hstack((distances, np.transpose([new_distances])))
    new_distances.append(0)
    distances = np.vstack((distances, [new_distances]))
    return clusters, distances