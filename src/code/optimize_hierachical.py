from geoopt import ManifoldParameter
from print_sphere import print_sphere
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from generate_pca_initial_guess import compute_center, normalize_points, get_results
from generate_data import get_flipkart
import torch, geoopt
import numpy as np
import torch.nn as nn
import pandas as pd

FIG_PATH = '../figures/hierachical_3dim_fig.png'
SAVE_PATH = '../data/Flipkart/hierachical_result.csv'

def print_stats():   
    from sklearn.metrics import pairwise_distances
    
    # May be really slow if you use a lot of points
    D = pairwise_distances(final_emb, metric='cosine')  # or 1–cosine
    np.fill_diagonal(D, np.inf)
    nn_dists = D.min(axis=1)
    print(np.mean(nn_dists))
    n = len(final_emb)
    similarities = final_emb @ final_emb.T
    print((np.sum(similarities) - n) / (n * (n - 1)))
    pos_sims = []
    neg_sims = []
    for p,c in edge_list:
        pos_sims.append(final_emb[p] @ final_emb[c])
        k = np.random.choice([i for i in range(n) if i not in (p,c)])
        neg_sims.append(normed[p] @ final_emb[k])

    print("Mean pos sim:", np.mean(pos_sims))
    print("Mean neg sim:", np.mean(neg_sims))

def index_tree_internal(root):
    '''
        Indexes a tree using a given root. Returns the internal reprsentation, indexed nodes, and edges in the tree.
    '''
    # Retrieve all nodes
    all_nodes = root.get_all_nodes(depth=0, exclude_leaves=True)
    internal = [(n, d) for n, d in all_nodes if n.parent is not None]   # We exclude root
    node2idx = {n: i for i, (n, _) in enumerate(internal)}

    edges = []
    for n, _ in internal:
        for c in n.children:
            if c in node2idx:
                edges.append((node2idx[n], node2idx[c]))
    return internal, node2idx, edges

class EdgeDataset(Dataset):
    def __init__(self, edges, neg_cands):
        self.edges     = edges
        self.neg_cands = neg_cands
        self.neg       = neg_samples
    def __len__(self):
        return len(self.edges)
    def __getitem__(self, idx):
        i, j = self.edges[idx]
        pool = self.neg_cands[i]
        pick = np.random.choice(pool, size=self.neg, replace=len(pool)<self.neg)
        return i, j, pick

# Get the data.
root, data, initial_guesses = get_flipkart()

# Check for cuda support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usining the device: ", device)

batch_size = 2048           # Number of edges processesd
neg_samples = 256           # Negative edges for loss
iterations = 100            # Number of iterations
pos_margin = 1.0            # Margin set for the hinge loss.
temperature = 0.8           # Decides how hard we push negatives. Lower = more push on negatives
dims = [128]                # Select the the dimension you want to optimize for.
print_statistics = False    # If you want to print positive and negative similarities between random parent and children.
lr = 1.0                    # Needs to adjusted for SGD and ADAM. SGD ~ 1.0 but can be higher and ADAM ~ 1e-3
w_min = 0                   # Set if wanted to ensure min distance between parent and child

all_nodes, node2idx, edge_list = index_tree_internal(root)
num_nodes = len(all_nodes)

print("Optimizing ", num_nodes, " product categoires.")


# Add parents and children as positives.
pos_nbrs = [[] for _ in range(num_nodes)]
for p, c in edge_list:
    pos_nbrs[p].append(c)
    pos_nbrs[c].append(p)
all_indexes = set(range(num_nodes))
neg_cands = [list(all_indexes - set(pos_nbrs[i]) - {i}) for i in range(num_nodes)]

parent_child_index = torch.tensor([p for p, _ in edge_list], device=device)
child_child_idx = torch.tensor([c for _, c in edge_list], device=device)
results = []

for d in dims:
    # PCA to initialize low dimensional embeddings and use the centering LS problem.
    orig_vecs = torch.stack([torch.as_tensor(n.embedding_vector, dtype=torch.float32) for n, _ in all_nodes])
    pca    = PCA(n_components=d)
    low_d   = pca.fit_transform(orig_vecs.numpy())
    estimated_center = compute_center(low_d)
    # Normalize the points with respect to the estimated center.
    optimal_points = normalize_points(low_d, estimated_center)
    emb_init = torch.tensor(optimal_points, dtype=torch.float32, device=device)

    # Create dataset
    dataset = EdgeDataset(edge_list, neg_cands)

    # Create dataloader to batch the data
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # We are working on the sphere
    manifold   = geoopt.SphereExact()
    embeddings = ManifoldParameter(emb_init, manifold=manifold)
    opt     = geoopt.optim.RiemannianSGD([{'params': embeddings}], lr=lr)
    loss_CEL = nn.CrossEntropyLoss()

    # Optimization loop
    for iteration in range(1, iterations+1):
        total_loss = 0.0
        for i_cpu, j_cpu, negs_cpu in loader:
            # Move to GPU
            i    = i_cpu.to(device, non_blocking=True)
            j    = j_cpu.to(device, non_blocking=True)
            negs = negs_cpu.to(device, non_blocking=True)


            # Preform cross entropy loss with hard negatives per building of weight matrix/datset.
            opt.zero_grad()
            hi, hj = embeddings[i], embeddings[j]
            hneg   = embeddings[negs]
            distance_pos  = manifold.dist(hi, hj)
            distance_neg = manifold.dist(hi.unsqueeze(1), hneg)
            input = torch.cat([-distance_pos.unsqueeze(1), -distance_neg], dim=1) / temperature
            target = torch.zeros(input.size(0), dtype=torch.long, device=device)
            loss_nce = loss_CEL(input, target)

            # Parent–child: enforce d(p,c) <= pos_margin
            if w_min > 0:
                p_emb   = embeddings[parent_child_index]
                c_emb   = embeddings[child_child_idx]
                d_hier  = manifold.dist(p_emb, c_emb)
                loss_min = torch.relu(pos_margin - d_hier).mean()
                loss = loss_nce + loss_min * w_min
            else:
                loss = loss_nce
            
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print("Iteration ", iteration, "/", iterations, "   Total_loss: ", total_loss)

    # Detach embeddings -> numpy
    final_emb = embeddings.detach().cpu().numpy()
    print(len(final_emb))

    names = [n.name for n,_ in all_nodes]

    # This should not be needed but is just a safety measure
    normed = final_emb / np.linalg.norm(final_emb, axis=1, keepdims=True)

    accuracy, ranked_kNN, score, lac = get_results(orig_vecs.detach().cpu().numpy(), normed, metric='cosine')
    results += [{"Accuracy": accuracy, "Ranked_kNN": ranked_kNN, "Points": len(normed), "Dim": d, "Score": score, "Lac": lac}]

    # We can print sphere if we are optimizing for taget_dim = 3
    if dims == 3:
        print_sphere(final_emb, FIG_PATH)

    if print_statistics:
        print_stats()


# Save the results to file.
frame = pd.DataFrame(results)
frame.to_csv(SAVE_PATH, index=False)