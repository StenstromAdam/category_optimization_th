# Repository for the thesis Scalable Optimization of product categories using Multi-Dimensional Scaling and LLM embeddings.

The required packages are located listed in requirements.txt and installed by running the following command in same folder as this file:
```
    pip install -r requirements.txt
```

The following files are included in the repository and corresponds to the following.

Currently all the program as are run with:

```
      python --FILE_NAME--
```

This will be README will be updated once that is fixed.


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