from umap import UMAP
import numpy as np

'''
A simple test for using UMAP witn normalization. Does not work, but was an explored idea.
'''

def optimize_umap(input_data, dim, neighbors=15):

    if(dim == 3):
        '''
            Taken from UMAP's website for utilizing haversine distance as metric. Works well but is a bit slow and only works for 3 dimensions.
        '''
        sphere_mapper = UMAP(metric='cosine', output_metric='haversine', n_neighbors=neighbors, low_memory=True).fit(input_data)

        x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
        y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
        z = np.cos(sphere_mapper.embedding_[:, 0])


        optimized_points = []

        x1, y1, z1 = list(x), list(y), list(z)

        for i in range(len(x1)):
            optimized_points.append([x1[i], y1[i], z1[i]])

        optimized_points = np.asarray(optimized_points)
        return optimized_points
    
    # Create mapper andr transform input data. Have to normalize after due to constraint.
    # A better solution would be to define output space that exactly fit our constraint. E.g sphere manifold etc.
    mapper = UMAP(metric='cosine', n_neighbors=neighbors, n_components=dim)
    optimized_points = mapper.fit_transform(input_data)
    optimized_points /= np.linalg.norm(optimized_points, axis=1, keepdims=True)

    return optimized_points



