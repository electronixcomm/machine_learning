import numpy as np
import pandas as pd

'''
Data
'''
data = np.matrix([[1,2,4], [4,1,2], [5,4,8]])
# data = np.matrix([[2.5, 2.4], [.5, 0.7], [2.2, 2.9],
#                   [1.9, 2.2], [3.1, 3.0], [2.3, 2.7],
#                   [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])
# data = np.matrix([[4, 11], [8, 4], [13, 5], [7, 14]])
# data = np.matrix([[1, 2], [2, 3], [3, 4], [4, 5]])
df = pd.DataFrame(data)

"""
From csv file
"""
df = pd.read_csv('data_file2.csv', header=None)
# df1  = df[:][:-1]
# print(df)

'''
Manual calculation
'''
print("Mean values:\n", df.mean())
# standardize data
# standardized_data = (df - df.mean()) / (df.std())
standardized_data = (df - df.mean()).values

# covnn = np.sum(standardized_data[:,1]*standardized_data[:,1])/10
# print(covnn)

# Finding covariance
covarance = np.cov(standardized_data.T)

# # find eigen value& eigen vector
eigenvalue, eigenvectors = np.linalg.eig(covarance)
# After taking PC1 eign vector arranging as eigen values in decreasing order if needed
# eigenvectors = np.flip(eigenvectors, axis=1)

# Find PCA
n_components = 2

pca_manual = np.dot(standardized_data, eigenvectors)

pca_manual = pca_manual[:, :n_components]


print('Standardized data')
print(standardized_data.round(2))
print('')

print('Covariance')
print(covarance.round(2))
print('')

print('eigen_value')
print(eigenvalue.round(4))
print('')


print('eigen_vector')
print(eigenvectors.round(4))
print('')

print('PCA calculated')
print(pca_manual.round(2))
print('')
