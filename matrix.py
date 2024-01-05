import numpy as np
c = np.diag([2] * 4)
# U1 = U[:, 0:K1]
# S1 = S[0:K1, 0:K1]
# V1 = V[:, 0:K1]
# Z1 = U1 * S1 * V1
u1 = c[:, 0:1]
s1 = c[0:1, 0:1]
v1 = c[:, 0:1]
print("u:", u1)
print("s", s1)
print("v", v1)

