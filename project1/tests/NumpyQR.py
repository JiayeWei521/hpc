import time
from numpy.linalg import qr
from helpers import matrix_1, loss_of_orthogonality

start_time = time.time()

m = 5000
n = 60
W =  matrix_1(m, n)
# W1 = test_matrix_1(m, n)

# QR factorization in Numpy library, based on Householder transformation
Q, R = qr(W)
loss = loss_of_orthogonality(Q)

end_time = time.time()

execution_time = end_time - start_time

print("Time taken = ", execution_time)
print("Q: \n", Q)
print("R: \n", R)
print("Loss of orthogonality = ", loss)