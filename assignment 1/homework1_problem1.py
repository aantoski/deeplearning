import numpy as np

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return np.dot(A,B) - C

def problem_1c (A, B, C):
    return A * B + np.transpose(C)

def problem_1d (x, y):
    return np.inner(x,y)

def problem_1e (A, i):
    return np.sum(A[i, 1::2])

def problem_1f (A, c, d):
    S = np.nonzero((A >= c) & (A <= d))
    nums_of_S = A[S]
    return np.mean(nums_of_S)

def problem_1g (A, k):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    sort = np.argsort(-np.abs(eigenvalues))
    return eigenvectors[sort[:k], :].T # return the k (count) eigenvalues

def problem_1h (x, k, m, s):
    n = x.shape[0]
    z = np.ones((n, 1)) # making a column vector (1xn) of all 1s
    I = np.eye(n) # .eye() is used to make the identity matrix
    mean = x + m * z
    cov = s * I
    flat_mean = mean.flatten() # this is done to make sure the mean is 1 dimensional
    return np.random.multivariate_normal(flat_mean, cov, size = k)

def problem_1i (A):
    new_column_order = np.random.permutation(A.shape[1])
    return A[:, new_column_order]

def problem_1j (x):
    mean = np.mean(x)
    standard_deviation = np.std(x)
    return (x - mean) / standard_deviation

def problem_1k (x, k):
    return np.repeat(x, k, axis=1)

def problem_1l (X, Y):
    X_three_dim = X[:, :, np.newaxis] # X to 3D
    Y_three_dim = Y[:, np.newaxis, :] # Y to 3D
    differences = X_three_dim - Y_three_dim # difference between each pair of vectors
    D = np.sqrt(np.sum(differences ** 2, axis=0))  # Find euclidean distance between each pair of vectors to convert to n x m
    return D

def run_all():
    print("Running all...")
    A = np.array([[1,2],[3,4]])
    A_big = np.array([[1,2,3,4], [1,2,3,4]])
    B = np.array([[1,3],[2,4]])
    C = np.array([[2,1],[1,2]])
    x = np.array([[1],[2],[3]])
    y = np.array([[4],[5],[6]])
    i = 1
    c = 2
    d = 3
    k = 2
    x_1h = np.array([[2],[3],[6],[8]])
    m = 1
    s = 2
    three_dim = np.array([[1,2,3],[4,5,6],[7,8,9]])
    X = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [4, 6, 2, 8]])
    Y = np.array([[9, 8],
                [6, 4],
                [3, 1]])
    
    print("Problem 1a:")
    print(problem_1a(A, B))
    print("Problem 1b:") 
    print(problem_1b(A, B, C))
    print("Problem 1c:") 
    print(problem_1c(A, B, C))
    print("Problem 1d:") 
    print(problem_1d(x,y))
    print("Problem 1e:") 
    print(problem_1e(A_big,i))
    print("Problem 1f:") 
    print(problem_1f(A,c,d))
    print("Problem 1g:") 
    print(problem_1g(A,k))
    print("Problem 1h:") 
    print(problem_1h(x_1h,k,m,s))
    print("Problem 1i:") 
    print ("Matrix A before:")
    print(three_dim)
    print("Matrix A after: ")
    print(problem_1i(three_dim))
    print("Problem 1j:") 
    print(problem_1j(three_dim))
    print("Problem 1k:") 
    print(problem_1k(x,k))
    print("Problem 1l:") 
    print(problem_1l(X,Y))
    

run_all()

