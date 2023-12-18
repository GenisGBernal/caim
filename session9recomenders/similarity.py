import numpy as np


def compute_similarity(l1: list, l2: list):
    # # @TODO: Implement the actual computation of similarity between vector1 and vector2.
    # The current implementation returns a placeholder value of 1. Update this function 
    # to perform the appropriate similarity calculation and return the result.

    ratings1 = [rating for rating in l1 if rating != -1.0]
    ratings2 = [rating for rating in l2 if rating != -1.0]

    list_ra = []
    list_rb = []
    for i in range(len(l1)):
        if l1[i] != -1.0 and l2[i] != -1.0:
            list_ra.append(l1[i])
            list_rb.append(l2[i]) 
    
    if len(list_ra) == 0: return 0.0

    # Convert lists to numpy arrays for easy computation
    ra = np.array(list_ra)
    rb = np.array(list_rb)

    # Calculate mean of each list
    mean_ra = np.mean(np.array(ratings1))
    mean_rb = np.mean(np.array(ratings2))

    # Calculate the difference vectors
    diff_ra = ra - mean_ra
    diff_rb = rb - mean_rb

    # Calculate the dot product and the denominators
    dot_product = np.sum(diff_ra * diff_rb)
    denominator_ra = np.sum(diff_ra**2)
    denominator_rb = np.sum(diff_rb**2)

    # Calculate the final result
    result = dot_product / (np.sqrt(denominator_ra) * np.sqrt(denominator_rb))

    return result

    # # Compute magnitudes
    # magnitude1 = np.linalg.norm(array1)
    # magnitude2 = np.linalg.norm(array2)
    # if magnitude1 == 0 or magnitude2 == 0:
    #     return 0

    # # Compute dot product
    # dot_product = np.dot(array1, array2)

    # # Compute cosine similarity
    # similarity = dot_product / (magnitude1 * magnitude2)

    # return similarity

if __name__ == "__main__":
    
    vector_a, vector_b = [0, 1], [1, 0]
    x = compute_similarity(vector_a, vector_b)
    print(x)