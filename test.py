import numpy as np
# a = [_ for _ in range(4096)]

# a = np.array(a)
# a = a.reshape(64,64)
# print(a)
# print(a[63].dot(a[]))

a = [0] * 64
b = [0] * 64
for i in range(64):
    a[i] = 4032 + i
    b[i] = 63 + 64 * i
    
a = np.array(a)
b = np.array(b)
c = a.dot(b)
print(c)