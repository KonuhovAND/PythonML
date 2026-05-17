import numpy as np

# NO LOOPS ALLOWED (FOR, WHILE, LIST COMPREHENSIONS) FOR ANY OF THESE TASKS!
# Use pure NumPy vectorized operations, broadcasting, and indexing methods.

# =====================================================================
# TASK 1: Vectorized Arithmetic & Broadcasting (From Conspectus 02 & 04)
# =====================================================================
# Given matrix X, perform the following element-wise operations:
# 1. Multiply all elements by 5.
# 2. Subtract the mean of each COLUMN from its respective column elements.
# 3. Add 10 to elements that are strictly greater than 0, leave others unchanged.

X = np.array([[-5,  12,   3],
              [ 0,  -4,   9],
              [ 7,   8,  -1],
              [-2,   0,   6]])

# TODO: Write your code for Task 1 below:
X*=5
X_avg =X.mean(axis=0)
X = X - X_avg
X[X>0] +=10


# =====================================================================
# TASK 2: Aggregations & Axis Operations (From Conspectus 04)
# =====================================================================
# Given the 2D array matrix 'A':
# 1. Find the maximum value in the entire matrix.
# 2. Find the index (row, column coords) of that maximum value using np.unravel_index.
# 3. Calculate the sum of elements across each ROW (axis=1).
# 4. Calculate the average value across each COLUMN (axis=0).

A = np.array([[ 2,  3,  0,  5],
              [ 8,  8,  0, 15],
              [ 0,  1,  6,  2]])

# TODO: Write your code for Task 2 below:
A_max=A.max()

A_max_index = np.unravel_index(np.argmax(A),A.shape)
# print(A_max_index[0],A_max_index[1])
A_sum_row = (A).sum(axis=1)
A_avg = A.mean(axis= 0)

# =====================================================================
# TASK 3: Boolean Indexing & Masks (From Conspectus 06)
# =====================================================================
# Given the array 'data':
# 1. Create a boolean mask where elements are even AND greater than 10.
# 2. Use this mask to select those elements into a new array.
# 3. Replace all elements in 'data' that are less than 0 with the number 999.

data = np.array([[ 12,  -3,   5,  14],
                 [  7,   0,  18,  -9],
                 [ 22,   4,  -1,  11]])

# TODO: Write your code for Task 3 below:

data_mask1 = np.logical_and((data>10),(data %2==0))
data_new = data[data_mask1]
data[data < 0] = 999
# print(data)


# =====================================================================
# TASK 4: Fancy Indexing & np.ix_ (From Conspectus 06)
# =====================================================================
# Given the matrix 'M':
# 1. Use Fancy Indexing/np.ix_ to extract a submatrix consisting of:
#    Rows: index 0 and index 2
#    Columns: index 1 and index 3
# 2. Change the values of this specific submatrix inside 'M' to 10000 
#    ensuring the original matrix 'M' is modified in-place.

M = np.arange(16).reshape(4, 4)
# [1,2,3,4].
# 5 6 7 8
# 9 10 11 12
# 13 14 15 16
# TODO: Write your code for Task 4 below:
new_M = M[np.ix_([0,2],[1,3])] 
M[np.ix_([0,2],[1,3])] = 10_000
# print(M)
print()
# print(new_M)
# =====================================================================
# TASK 5: Array Manipulation & Concatenation (From Conspectus 08)
# =====================================================================
# Given two arrays 'v1' and 'v2':
# 1. Reshape both into 2D column vectors (shape 4x1).
# 2. Concatenate them horizontally to form a 4x2 matrix.
# 3. Flatten the resulting 4x2 matrix back into a 1D array using .flatten()

v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 8])

# TODO: Write your code for Task 5 below:

v1=v1.reshape((4,-1))
v2 = v2.reshape((4,-1))
v3 =np.concatenate((v1,v2),axis=1) 
print(v3)
v3= v3.flatten()
print(v3)
# =====================================================================
# VERIFICATION / SANITY CHECKS
# Run your script in VS Code to see if your outputs print correctly.
# =====================================================================
print("--- Check Task 1 Results ---")
print(X)
print("\n--- Check Task 2 Results ---")
print(A_max)
print(A_max_index[0],A_max_index[1])
print("\n--- Check Task 3 Results ---")
# print(data)
print("\n--- Check Task 4 Results ---")
# print(M)
print("\n--- Check Task 5 Results ---")
# print(your_task_5_result)