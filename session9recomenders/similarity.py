import numpy as np


def compute_similarity(vector1: list, vector2: list):
    # # @TODO: Implement the actual computation of similarity between vector1 and vector2.
    # The current implementation returns a placeholder value of 1. Update this function 
    # to perform the appropriate similarity calculation and return the result.
    
    # Convert lists to NumPy arrays
    array1 = np.array(vector1)
    array2 = np.array(vector2)

    
    # Compute magnitudes
    magnitude1 = np.linalg.norm(array1)
    magnitude2 = np.linalg.norm(array2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    # Compute dot product
    dot_product = np.dot(array1, array2)

    # Compute cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity

if __name__ == "__main__":
    
    vector_a, vector_b = [0, 1], [1, 0]
    x = compute_similarity(vector_a, vector_b)
    print(x)