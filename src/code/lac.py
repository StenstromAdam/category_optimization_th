import numpy as np
from sklearn.neighbors import NearestNeighbors

def get_results_(input, optimal, neighbors=15, metric='cosine'):
    '''
        Find the least accurate component (LAC) for a given input, with a given optimal solution, checking for neighbors
        that match both input and output space, using the metric.
    '''

    # Find the nearest neighbors for optimal and input space.
    optimal_neighbors = NearestNeighbors(n_neighbors=neighbors, metric=metric, n_jobs=-1).fit(optimal)
    input_neighbors = NearestNeighbors(n_neighbors=neighbors, metric=metric, n_jobs=-1).fit(input)

    # Return the indices of nearest neighbors for each space (excluding itself).
    o = optimal_neighbors.kneighbors(return_distance=False)
    i = input_neighbors.kneighbors(return_distance=False)

    # Calcualte a score based on the number of overlapping matches between the two sets.
    unranked = np.array([len(set(row_A) & set(row_B)) for row_A, row_B in zip(i, o)])
    
    ranked = np.sum(o == i)

    ranked = ranked / (neighbors * len(input))
    
    # The one index with the least amount of overlapping nearest neighbors.
    lac_index = np.argmin(unranked)

    unranked = unranked.sum() / (neighbors * len(unranked))
    return ranked, unranked, lac_index