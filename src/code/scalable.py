from category_optimization_th.src.code.generate_pca_initial_guess import get_results
from pytorch_metric_learning import miners, losses
from sklearn.neighbors import NearestNeighbors
from geoopt.optim import RiemannianSGD, RiemannianAdam
from geoopt.manifolds import SphereExact, Sphere
from geoopt import ManifoldParameter
from generate_pca_initial_guess import compute_center, normalize_points
from sklearn.decomposition import PCA
from print_sphere import print_sphere
from torch.utils.data import TensorDataset, dataloader
from tqdm import tqdm
import torch
import numpy as np
import faiss

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X: torch.Tensor, U: torch.Tensor, S: torch.Tensor):
        self.X = X
        self.U = U
        self.S = (S**2)
        self.N, self.D = X.shape
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        x = self.X[idx]
        # Computes the weights
        proj = self.U.T @ x
        scaled = self.S * proj
        w = self.U @ scaled
        return x, w

def cost_stress(X, W, knn_index, eps=1e-8):
    cos_ij = (X.unsqueeze(1) * knn_index).sum(-1)
    cos_ij = cos_ij.clamp(-1 + eps, 1 - eps)
    d_ij   = torch.acos(cos_ij)
    return torch.nn.functional.mse_loss(d_ij, W)

def cost_hinge(X, W, N, knn_index, k=15, hard_neg_ration=2):
    vec_matrix = torch.arange(N, device=device).unsqueeze(1).expand(N, k)
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


PATH_EMBEDDINGS = "../data/Amazon/amazon_corpus_embeddings.npy"

# Check for cuda support. Will greatly increase the speed.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", 'cuda' if torch.cuda.is_available() else 'cpu')

data = np.load(PATH_EMBEDDINGS, mmap_mode='r')[:100000].astype('float32')

# Number and Dimensions
N, D = data.shape
data_torch = torch.from_numpy(data)


# Variables
target_dim = 128
iterations = 1000
W_stress = 1.0
W_hinge = 1.0
eps = 1e-7
lr = 1e-3
k = 15
batch_size = 4096
solver = 'Adam'
weight_stress = 10.0
weight_hinge = 1.0

# Generate initial guess through PCA
pc = PCA(n_components=target_dim).fit_transform(data)
center = compute_center(pc)
X_init = normalize_points(pc, center)
X_init = torch.from_numpy(X_init).float()
#print("Mean Cosine:", (data.sum(axis=0).dot(data.sum(axis=0)) - np.einsum("ij,ij->i", data, data).sum()) / (len(data) * (len(data) - 1)))


# Our Dataset
#dataset = TensorDataset(X_init)

# Solver and manifold
manifold = SphereExact()
parameters = ManifoldParameter(X_init.clone().detach().to(device).requires_grad_(True),
                                manifold=manifold
                                ).cuda(device=device)

# Initialize optimizer with given parameters and learn rate.

match solver:
    case 'Adam':
        optimizer = RiemannianAdam([parameters], lr=lr)
    case 'SGD':
        optimizer = RiemannianSGD([parameters], lr=5000*lr, momentum=0.9)
    case _:
        print("No specified or worng solver. Needs to be Adam or SGD.")
        exit(0)


# Precalculate lables or kNN
res = faiss.StandardGpuResources()
gpu_index = faiss.GpuIndexFlatL2(res, D)
gpu_index.add(data)
dists_hd, indx = gpu_index.search(data, k+1)
knn_indices = torch.tensor(indx[:, 1:], device=device) # We remove the first since it's always itself
dists_hd = torch.tensor(dists_hd[:,1:], device=device)
dists_hd = 1.0 - 0.5 * dists_hd
dists_hd = torch.clamp(dists_hd, -1 + eps, 1 - eps)
dists_hd = torch.acos(dists_hd)
#dists_hd = torch.acos(torch.clamp(1 - torch.tensor(dists_hd[:,1:], device=device), -1+eps,1-eps))
U, S, Vh = torch.linalg.svd(dists_hd, full_matrices=False)
def gram(v: torch.Tensor) -> torch.Tensor:
    proj = U.T @ v
    s = (S**2).unsqueeze(-1) * proj if v.ndim == 2 else (S**2) * proj
    return U @ s

X_gpu = X_init.to(device=device)
W_full = dists_hd.to(device)
knn = knn_indices.cpu()

#dataset = TensorDataset(X_init, W_full, knn)


ids = torch.arange(parameters.shape[0], dtype=torch.long)
# Our data loader for batching
loader = torch.utils.data.DataLoader(
    TensorDataset(ids),
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)

# Precompute kNN

# Training Loop
for iteration in tqdm(range(iterations)):
    for (batch_idx, ) in loader:
        batch_idx = batch_idx.to(device, non_blocking=True)
        batch_emb = parameters[batch_idx]
        batch_w = W_full[batch_idx]
        batch_knn = knn_indices[batch_idx]
        n_e = parameters[batch_knn]        
        loss_stress = cost_stress(batch_emb, batch_w, batch_knn)
        #loss_hinge = cost_hinge(batch_e, batch_w, knn_indices, N=batch_size, k=k)
        loss_total = weight_stress * loss_stress #+ weight_hinge * loss_hinge
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()




optimal_points = parameters.detach().cpu().numpy()

if target_dim == 3:
    print_sphere(optimal_points, "gz.png")

print(get_results(data, optimal_points))