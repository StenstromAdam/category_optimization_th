from generate_data import get_amazon, get_flipkart, get_restaurants  # Used for generation tree and fetching data
from joblib import Parallel, delayed
from tqdm import tqdm
from generate_pca_initial_guess import get_results
from sklearn.manifold import smacof
from sklearn.metrics import pairwise_distances

# Optimization functions
#from proj_opt_categories import projected_gradient
#from optimize_umap import opt_umap
from optimize_smacof import opt_smacof

import time
import numpy as np
import pandas as pd

def opt_smacof(initial_x, input_data, dim):
    # Pariwise distances returns cosine dissimilarities
    dissimilarities = pairwise_distances(input_data, metric='cosine')
    # Enforce symmetry due to rounding errors, otherwise d.T = d
    dissimilarities = (dissimilarities + dissimilarities.T) / 2.0
    mds_result, stress = smacof(dissimilarities, n_components=dim, init=initial_x, n_jobs=-1, max_iter=300)
    mds_result /= np.linalg.norm(mds_result, axis=1, keepdims=True)
    return mds_result

'''
def run_pg_for_dimensions(initial_x, input_data, dim, optima):

    start = time.time()
    optimal_points = projected_gradient(input_data, initial_x)
    elapsed = time.time() - start
    accuracy, accuracy_r, score, lac = get_results(input_data, optimal_points)
    optima.append(optimal_points)
    # REMOVE [] FOR PARALLELL :-| I don't know why
    return {"Ranked_Accuracy": accuracy_r, "Accuracy": accuracy, "Time": elapsed, "Points": points, "Dim": dim, "Score": score, "Lac": lac}

def run_umap(input_data, dim, optima):
    start = time.time()
    optimal_points = opt_umap(input_data, dim)
    elapsed = time.time() - start
    accuracy, score, lac = get_results(input_data, optimal_points, metric='cosine')
    optima.append(optimal_points)
    return [{"Accuracy": accuracy, "Time": elapsed, "Points": points, "Dim": dim, "Score": score, "Lac": lac}]
'''

def run_smacof(initial_x, input_data, dim, optima, points):
    start = time.time()
    optimal_points = opt_smacof(initial_x, input_data, dim)
    elapsed = time.time() - start
    accuracy, score, lac = get_results(input_data, optimal_points)
    optima.append(optimal_points)
    return [{"Accuracy": accuracy, "Time": elapsed, "Points": points, "Dim": dim, "Score": score, "Lac": lac}]



if __name__ == '__main__':
    
    dataset = 'Flipkart'
    method = 'smacof'       # Only one available, although others can easily be implemented by undoing the comments and some fixes.
    dimensions = [128]      # Target dimensions to optimize for.
    
    # Retrieve hiearchical tree and data from dataset
    match dataset:
        case 'Flipkart':
            tree, data, initial_guesses = get_flipkart()
        case 'Restaurants':
            data, initial_guesses = get_restaurants()
        case 'Amazon':
            data, initial_guesses = get_amazon()
        case _:
            print("No such dataset.")
            exit(0)
            pass
    


    # Used to store results and optimal points if using more than 1 dimensions
    input_data = list(data.values())
    results = []
    optima_matrices = []

    match method:
        case 'projected_gradient': # Standard gradient based method
            '''
            results += Parallel(n_jobs=10)(
                delayed(run_pg_for_dimensions)(
                    initial_guesses[x],
                    input_data,
                    dim,
                    optima_matrices,
                ) for dim, x in tqdm(zip(dimensions, initial_guesses))
            )
            '''
        case 'smacof':
            for dim, x in tqdm(zip(dimensions, initial_guesses)):
                results += run_smacof(
                    initial_guesses[x],
                    input_data,
                    dim,
                    optima_matrices,
                    len(data),
                )
        case 'umap':
            '''
            for dim in tqdm(range):
                results += run_umap(
                    input_data,
                    dim,
                    optima_matrices
                )
            '''
        case _:
            pass

    frame = pd.DataFrame(results)
    frame.to_csv('../evaluation/' + dataset + '/' + method + '_' + dataset + '.csv', index=False)
    np.savez('../evaluation/' + dataset + '/' + method + '_' + dataset + '.gz', *optima_matrices)
