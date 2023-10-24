import time
from numpy.linalg import qr, cond
from helpers import matrix_1, loss_of_orthogonality, test_matrix_1

m = 50000
n = 600
W =  test_matrix_1(m, n)
# W1 = test_matrix_1(m, n)

start_time = time.time()
# QR factorization in Numpy library, based on Householder transformation
Q, R = qr(W)
end_time = time.time()

execution_time = end_time - start_time

print("Time taken = ", execution_time)
# print("Q: \n", Q)
# print("R: \n", R)

print("Loss of orthogonality: ", loss_of_orthogonality(Q))
print("Condition number: ", cond(Q))