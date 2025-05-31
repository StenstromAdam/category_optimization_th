import re, os, torch
import numpy as np
import pandas as pd
import numpy as np
from datasets import load_from_disk, Dataset

FIGURE_PATH = '../figures/sphere_plot'
RESULT_PATH = '../evaluation/result.txt'
EMBED_PATH = '../data/Flipkart/text_embeddings_mxbai-embed-large-v1.pt'
DATASET_PATH = '../data/Flipkart/flipkart.csv'
EMBED_PATH_RES = '../data/Restaurants/text_embeddings_mxbai-embed-large-v1.pt'
DATASET_PATH_RES = '../data/Restaurants/restaurants.csv'

class Node:
    def __init__(self, name, embedding_vector):
        self.name = name
        self.embedding_vector = embedding_vector
        self.children = []
        self.parent = None

    def addChild(self, child):
        child.parent = self
        self.children.append(child)
    
    def addEmbeddingVector(self, embedding_vector):
        self.embedding_vector = embedding_vector

    def get_all_nodes(self, depth=0, exclude_leaves=True):
        """
        Traverse the tree and return all nodes with their depths, excluding leaves if specified.
        """
        nodes = []
        if not exclude_leaves or len(self.children) > 0:  # Include non-leaf nodes only
            nodes.append((self, depth))
        for child in self.children:
            nodes.extend(child.get_all_nodes(depth + 1, exclude_leaves))
        return nodes
    
def check_file(path):
    '''
    Checks if the specified path exists, exits program if not.
    '''
    if not os.path.isfile(path):
        print("Path: ", path, "\n Not found. Please make sure the dataset exists")
        exit()
    return

    
def get_flipkart(file_path=EMBED_PATH, csv_path=DATASET_PATH):
    '''
    Retrieves the flipkart dataset and creates a hierachical tree from the data.
    returns the root of the tree, a dictionary with all the categories and embeddings, as well as an inital guess matrix if it exists.
    '''
    # Read the data from the csv file and embedding vectors as a dataset
    ds = read_data_flip(file_path, csv_path)

    # Create root of hierachical tree.
    root = Node('root', None)

    # Go through the hierachical text and add them to tree.
    j = 0
    for category_hierarchy_text in ds['product_category_tree']:
        # Regex to find all the categories in the product category tree, needs to be adjusted if other structre than specified in report.
        list = re.findall("([A-Z,a-z,0-9,@][^>]*)(?:[\" ])", category_hierarchy_text)

        # Add the list of products from the regex search to the root.
        add_list_to_root(list, root, 0, ds['embedding_vectors'][j])
        j += 1
    print(j)
    category_dictionary = {}
    _, category_dictionary = calculate_embedding_vectors(root, category_dictionary)

    # Remove root from tree since it does not represent anything.
    category_dictionary.pop('root')

    try:
        initial_guesses = np.load("../evaluation/flip/PCA_INPUT_FLIP.gz.npz")
    except Exception as e:
        print("File not found. A input file needs to be created using the \"create_input\" file. \n\n")
        print(e)
        initial_guesses = None

    return root, category_dictionary, initial_guesses

def get_restaurants(file_path=EMBED_PATH_RES, csv_path=DATASET_PATH_RES):

    check_file(csv_path)

    dataset_rest = read_data_restaurants(file_path, csv_path)
    cat_dict = {}
    i = 0
    for d in dataset_rest['Restaurant Name']:
        cat_dict.update({d : dataset_rest['embedding_vectors'][i]/np.linalg.norm(dataset_rest['embedding_vectors'][i])})
        i += 1

    try:
        initial_guesses = np.load("../evaluation/rest/PCA_INPUT_REST.gz.npz")
    except Exception as e:
        initial_guesses = None
        print("File not found. A input file needs to be created using the \"create_input\" file. \n\n")
        print(e)
        #exit(0)
    return cat_dict, initial_guesses

def read_data_flip(embedding_path, csv_path):

    check_file(embedding_path)

    # Load value of embeddings vectors from the file specifeid in FILE_PATH
    embedding_vectors = torch.load(embedding_path, weights_only=False)

    # Load the csv file into a dataset
    csv_file = pd.read_csv(csv_path, usecols=['product_name', 'product_category_tree'])

    # Add the embedding vectors to the dataset
    csv_file['embedding_vectors'] = list(embedding_vectors)

    return csv_file

def read_data_restaurants(embedding_path, csv_path):
    if not os.path.isfile(embedding_path):
        print("Wrong file path. Edit row 6 of make_tree.py and make the file exists")
        exit()

    # Load value of embeddings vectors from the file specifeid in FILE_PATH
    embedding_vectors = torch.load(embedding_path, weights_only=False)
    embedding_vectors = embedding_vectors.tolist()

    #print(len(embedding_vectors))

    # Load the csv file into a dataset
    csv_file = pd.read_csv(csv_path, usecols=['Restaurant Name'], nrows=len(embedding_vectors))

    # Add the embedding vectors to the dataset
    csv_file['embedding_vectors'] = embedding_vectors

    return csv_file


def add_list_to_root(lst, current, i, embedding_vector):

    # Check if we've reached the end of the list; return
    if i >= len(lst):
        return
    
    child = next((child for child in current.children if child.name == lst[i]), None)
    
    # Check if child is in tree already
    if child is None or (child is not None and i == len(lst) - 1):
        # Create a new node
        child = Node(lst[i], embedding_vector)
        current.addChild(child)

    # Recurisve call to add all remaining nodes to list
    add_list_to_root(lst, child, i + 1, embedding_vector)

def calculate_embedding_vectors(node, cat_dict):
    # Leaf => return embedding vector
    if len(node.children) == 0:
        return node.embedding_vector, cat_dict
    
    temp_ev = torch.zeros(1024)

    for child in node.children:
        ev, cat_dict = calculate_embedding_vectors(child, cat_dict)
        temp_ev += ev

    temp_ev = temp_ev/len(node.children)

    temp_ev = temp_ev/np.linalg.norm(temp_ev)
    
    node.addEmbeddingVector(temp_ev)
    # Temporary solution for having multiple of the same categories eventhough they are at different places in the hierachy.
    for nbr in ["", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        if node.name + nbr not in cat_dict:
            cat_dict.update({node.name + nbr: node.embedding_vector})
            break
    cat_dict.update({node.name: node.embedding_vector})
    return node.embedding_vector, cat_dict

def get_amazon():
    '''
        Returns the dataset for amazon.
    '''
    dataset = load_from_disk("../data/Amazon2")  # adjust path if needed
    data = dataset.to_pandas().to_dict()
    data = dict(zip(data["category_name"].values(), data["average_embedding"].values()))
    try:
        intial_guesses = np.load("../evaluation/amaz/PCA_INPUT_AMAZ.gz.npz")
    except:
        intial_guesses = None
    return data, intial_guesses