import numpy as np


def compute_similarity(l1: list, l2: list):
    # # @TODO: Implement the actual computation of similarity between vector1 and vector2.
    # The current implementation returns a placeholder value of 1. Update this function 
    # to perform the appropriate similarity calculation and return the result.

    vec1 = np.array(l1)
    vec2 = np.array(l2)

        # Find indices where both vectors have valid ratings
    valid_indices = np.where((vec1 != -1) & (vec2 != -1))[0]
    
    # Extract valid ratings from both vectors
    vec1_valid = vec1[valid_indices]
    vec2_valid = vec2[valid_indices]

    # Subtract mean from each vector
    vec1_valid -= np.mean(vec1_valid)
    vec2_valid -= np.mean(vec2_valid)

    # Calculate cosine similarity
    dot_product = np.dot(vec1_valid, vec2_valid)
    magnitude_vec1 = np.linalg.norm(vec1_valid)
    magnitude_vec2 = np.linalg.norm(vec2_valid)

    # Handle division by zero
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    else:
        similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
        return similarity
    
    
if __name__ == "__main__":
    
    vector_a, vector_b = [0, 1], [1, 0]
    x = compute_similarity(vector_a, vector_b)
    print(x)