import torch, geoopt, faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from geoopt.optim import RiemannianAdam, RiemannianSGD
from generate_pca_initial_guess import get_results, compute_center, normalize_points
from sklearn.decomposition import PCA
from tqdm import tqdm
from generate_data import get_flipkart, get_restaurants, get_amazon


def get_lac(input, optimal, neighbors=15):
    '''
        Find the least accurate category (LAC) for a given input, with a given optimal solution, checking for neighbors
        that match both input and output space. Used instead of trustworthiness due to time constraints.
    '''

    res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexFlatL2(res, input.shape[1])
    gpu_index.add(input)
    _, i = gpu_index.search(input, k+1)

    # Precalculate lables or kNN
    res_ld = faiss.StandardGpuResources()
    gpu_index_ld = faiss.GpuIndexFlatL2(res_ld, optimal.shape[1])
    gpu_index_ld.add(optimal)
    _, o = gpu_index_ld.search(optimal, k+1)

    i = i[:,1:]
    o = o[:,1:]

    ranked = np.sum(i == o)

    ranked = ranked / (neighbors * len(input))

    # Calcualte a score based on the number of overlapping matches between the two sets.
    unranked = np.array([len(set(row_A) & set(row_B)) for row_A, row_B in zip(i, o)])
    
    # The one index with the least amount of overlapping nearest neighbors.
    lac_index = np.argmin(unranked)

    unranked = unranked.sum() / (neighbors * len(unranked))
    return ranked, unranked, lac_index

if __name__ == '__main__':
    
    dataset = 'Flipkart'

    match dataset:
        case 'Flipkart':
            tree, data, initial_guesses = get_flipkart()
            data = np.ascontiguousarray(list(data.values()), dtype=np.float32)
        case 'Restaurants':
            data, initial_guesses = get_restaurants()
            data = np.ascontiguousarray(list(data.values()), dtype=np.float32)
        case 'Amazon':
            data, initial_guesses = get_amazon()
            data = np.ascontiguousarray(list(data.values()), dtype=np.float32)
        case 'Big':
            PATH_EMBEDDINGS = "../data/Amazon/amazon_corpus_embeddings.npy"
            data = np.load(PATH_EMBEDDINGS, mmap_mode='r')[:10000].astype('float32')
        case _:
            print("No such dataset.")
            exit(0)
            pass
 

    # Learn rate and number of iterations
    k = 15
    dims = [16, 32, 48, 64, 82, 100, 114, 128]
    solver = 'SGD'
    lr = 1e-3
    iterations = 2000
    neg_ratio = 5
    hard_neg_ration = 2
    stress_weight = 6.0
    hinge_weight = 3.0
    eps = 1e-7

    # Vectors for storing results.
    results = []
    optima_matrices = []

    # Check for cuda support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", 'cuda' if torch.cuda.is_available() else 'cpu')

    # Precompute true kNN and index them
    X_hd = np.vstack(data)
    N = X_hd.shape[0]
    res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexFlatL2(res, data.shape[1])
    gpu_index.add(data)
    dists_hd, inds_hd = gpu_index.search(data, k+1)
    # We remove the first index due to being itself
    knn_index = torch.tensor(inds_hd[:,1:], dtype=torch.long, device=device)

    # High dimensional geodesic angles (same as acos) with eps for non-inf values 
    d2 = torch.tensor(dists_hd[:, 1:], device=device)
    dists_hd = torch.acos(torch.clamp(1 - d2/2.0, -1+eps, 1-eps))

    for dim in dims:

        # ||x||= 1 constraint is upheld on the sphere manifold
        manifold = geoopt.SphereExact()

        # Inital guess for dim
        X0 = PCA(n_components=dim).fit_transform(X_hd)
        center = compute_center(X0)
        X0 = normalize_points(X0, center)

        # Initiziate the geoopt manifold given the initial guess as a torch tensor using our device
        X = geoopt.ManifoldParameter(torch.tensor(X0, dtype=torch.float32, device=device), manifold=manifold)
        
        # Initialize solver
        if solver == 'SGD':
            optimizer = RiemannianSGD([X], lr=lr*1000, momentum=0.9)
        else:
            optimizer = RiemannianAdam([X], lr=lr)
            

        for iteration in tqdm(range(1, iterations+1)):

            # Reset the gradients
            optimizer.zero_grad()

            # Create matrix so we can vectorize the computation
            vec_matrix = torch.arange(N, device=device).unsqueeze(1).expand(N, k)
            Xi = X[vec_matrix]      # Negatives
            Xj = X[knn_index]       # Positives

            # We calcualte the geodesic distances and stress loss from target
            cos_ij = (Xi * Xj).sum(dim=2).clamp(-1+eps,1-eps)
            d_ij = torch.acos(cos_ij)
            stress_loss = ((d_ij - dists_hd)**2).mean()

            # Our loss is the total stress times the set weight
            tot_stress = (stress_weight * stress_loss)

            vec_matrix_flat = vec_matrix.reshape(-1)
            pos_flat = knn_index.reshape(-1)

            neg_pool = torch.randint(0, N, (vec_matrix_flat.size(0), hard_neg_ration * 4), device=device)

            # Compute distances for selection
            Xa_flat = X[vec_matrix_flat].unsqueeze(1).expand(-1, hard_neg_ration * 4, -1)
            Xn_pool = X[neg_pool]
            cos_n_pool = (Xa_flat * Xn_pool).sum(dim=2).clamp(-1+eps,1-eps)
            distances_negative = torch.acos(cos_n_pool)

            # Select smallest distances
            _, indexes_hard = torch.topk(-distances_negative, hard_neg_ration, dim=1)

            # positive distances flattened
            cos_positives_flat = (X[vec_matrix_flat] * X[pos_flat]).sum(dim=1).clamp(-1+eps,1-eps)
            distances_positive_flat = torch.acos(cos_positives_flat)
            distances_positive_exp = distances_positive_flat.unsqueeze(1).expand(-1, hard_neg_ration)
            # select negative distances
            distances_negative_hard = torch.gather(distances_negative, 1, indexes_hard)

            # Hinge violations with RELU
            viol = torch.relu(distances_positive_exp - distances_negative_hard)
            hinge_loss = viol.mean()

            # combined hinge + stress
            total_loss = hinge_weight * hinge_loss + tot_stress
            total_loss.backward()

            optimizer.step()

        # Final evaluation
        optimized_points = X.detach().cpu().numpy()
        
        
        if dataset == 'Big':
            ranked, unranked, lac = get_lac(X_hd, optimized_points)
            results += [{"Unraked": unranked, "Ranked": ranked, "Points": len(optimized_points), "Dim": dim, "Lac": lac}]
        else:
            trust, unranked, ranked, lac = get_results(X_hd, optimized_points)
            results += [{"Trustworthiness": trust, "Ranked_kNN": ranked,  "Points": len(optimized_points), "Dim": dim, "Ranked": unranked, "Lac": lac}]

        optima_matrices.append(optimized_points)

    # Save result
    frame = pd.DataFrame(results)
    frame.to_csv('../evaluation/' + dataset + '/' + solver + '_' + dataset + '.csv', index=False)
    np.savez('../evaluation/' + dataset + '/' + solver + '_' + dataset + '.gz', *optima_matrices)