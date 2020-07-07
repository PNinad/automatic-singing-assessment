import numpy as np
import modular_dtw
# from scipy.spatial.distance import euclidean
import distance_metrics
a = np.array([1,2,3])
b = np.array([1,2,3,4])
dist_func = distance_metrics.euclidean
cost_matrix = modular_dtw.cost_matrix(a, b, dist_func)

print(cost_matrix)

path = modular_dtw.path(cost_matrix)
print(path)
