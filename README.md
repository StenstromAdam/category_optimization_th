# Repository for the thesis Scalable Optimization of Product Categories using Multi-Dimensional Scaling and LLM embeddings.

The required packages are located in requirements.txt and installed by running the following command in same folder as this file:
```
    pip install -r requirements.txt
```

Currently all the program as are run with:

```
      python --FILE_NAME--
```

However, another solution will be implemented and this README will be updated once that is fixed.


The following files are included in the repository and corresponds to the following.
```
category_optimization_th
|   README.md
|   requirements.txt
|
└─src
  |     
  └─────data      # All the data for any datset is stored here
        |
        └─ Amazon                   
        └─ Flipkart
        └─ Restaurants
  |     
  └─────code      # Code for different optimizations
        |
        └─  generate_data.py                    # Helper functions for generating tree strucutre and fetching datasets.
        └─  generate_pca_initial_guess.py       # Generates the intial guess matrices that are used for inputs to the other methods. 
        └─  lac.py                              # Function for getting results               
        └─  optimize_hierachical.py             # Optimiziation for hierachical structure
        └─  optimize_smacof.py                  # Optimize for smacof
        └─  optimize_manifold.py                # Optimize using riemannian ADAM and SGD
        └─  print_sphere.py                     # Function for printing sphere/cricle for 2 and 3 dimensions.
  |     
  └─────evaluation      # Any file generated from code is stored/put here.
        |
        └─ Amazon           
        └─ Flipkart
        └─ Restaurants
  |
  └─────exlpored_ideas    # Some explored ideas we tried
        |
        └─optimize_umap.py
        └─optimize_projected_gradient.py            
```