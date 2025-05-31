from sklearn.manifold import trustworthiness # Used to compare the initial points with the given resulting points for accuracy
from generate_data import get_flipkart, get_restaurants, get_amazon # Used for generation tree and fetching data
from lac import get_results_
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import time
import numpy as np
import pandas as pd

from numpy.linalg import lstsq

def compute_center(points):
    A = 2 * (points)
    b = np.sum(points**2, axis=1)
    center, residuals, rank, singular_values = lstsq(A, b, rcond=None)
    return center

def normalize_points(points, center):
    shifted_points = points - center
    norms = np.linalg.norm(shifted_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_points = shifted_points / norms
    return normalized_points

def print_nearest_for_point(optimal_points, data, index, neighbour_accuracy=15):
    nbre = NearestNeighbors(n_neighbors=neighbour_accuracy, metric='cosine', algorithm='brute', n_jobs=-1).fit(optimal_points)
    o = nbre.kneighbors(return_distance=False)

    nbrez = NearestNeighbors(n_neighbors=neighbour_accuracy, metric='cosine', algorithm='brute', n_jobs=-1).fit(list(data.values()))
    p = nbrez.kneighbors(return_distance=False)

    print("First", list(data.keys())[index])
    i = 1
    for z in o[index]:
        print(i, list(data.keys())[z])
        i += 1

    print("First", list(data.keys())[index])
    i = 1
    for z in p[index]:
        print(i, list(data.keys())[z])
        i += 1

def get_results(input_data, optimal_points, neighbour_accuracy=15, metric='cosine'):
    '''
    Returns trustworthiness, ranked unranked, lac
    '''
    trust = trustworthiness(input_data, optimal_points, n_neighbors=neighbour_accuracy, metric=metric)
    ranked, unranked, lac_index = get_results_(input_data, optimal_points, neighbors=neighbour_accuracy, metric=metric)
    return trust, ranked, unranked, lac_index


def calculate_pca_input(input_data, points, dimensions):
    '''
    Returns the dataset for results
    '''
    optima_matrices = []
    results = []
    for dim in dimensions:
        start_time = time.time()
        pca = PCA(n_components=dim)
        optimal_points = pca.fit_transform(input_data)  
        estimated_center = compute_center(optimal_points)
        # Normalize the points with respect to the estimated center.
        optimal_points = normalize_points(optimal_points, estimated_center)
        optimal_points /= np.linalg.norm(optimal_points, axis=1, keepdims=True)
        e_time = time.time() - start_time
        accuracy, score, lac = get_results(input_data, optimal_points)
        results += [{"Accuracy": accuracy, "Time": e_time, "Points": points, "Dim": dim, "Score": score, "Lac": lac}]
        optima_matrices.append(optimal_points)
    return pd.DataFrame(results), optima_matrices


if __name__ == "__main__":

    dataset = 'Flipkart'    # Chosen dataset
    dimensions = [128]      # Select number of dimensions to optimzie for

    # Retrieve hiearchical tree and data from dataset
    match dataset:
        case 'Flipkart':
            tree, data, initial_guesses = get_flipkart()
        case 'Restaurants':
            data, initial_guesses = get_restaurants()
        case 'Amazon':
            data, _ = get_amazon()
        case _:
            pass

    list_of_frames = []

    result_frame, optima_matrices = calculate_pca_input(list(data.values()), len(data), dimensions)
    list_of_frames.append(result_frame)

    avg_frame = sum(df.drop(columns=['Lac']) for df in list_of_frames) / len(list_of_frames)
    avg_frame['Lac'] = list_of_frames[0]['Lac']

    # Save the dataset and values into the evaluation folders
    avg_frame.to_csv('../evaluation/' + dataset + '/PCA_' + dataset + '.csv', index=False)
    np.savez('../evaluation/' + dataset + '/PCA_' + dataset + '.csv', *optima_matrices)




