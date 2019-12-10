import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

X = np.array(df.Value)
N = len(X)
C = 4
m = 2.1
e = 0.01

U_init = np.random.rand(C, N)
v_init = np.zeros(C)

for j in range(N):
    temp = 0
    for i in range(C):
        temp += U_init[i][j]
    for i in range(C):
        U_init[i][j] = U_init[i][j]/temp

def v_update(U):
    v = np.zeros(C)
    for i in range(C):
        tmp = 0
        t = 0
        for k in range(N):
            t += U[i][k]**m * X[k]
        for k in range(N):
            tmp += U[i][k]**m
        v[i] = t/tmp
    return v

def U_update(U, v):
    U_upd = np.matrix.copy(U)
    for i in range(C):
        for k in range(N):
            tmp = 0
            for j in range(C):
                tmp += (abs(X[k] - v[i])/abs(X[k] - v[j]))**(2/(m-1))
            U_upd[i][k] = 1/tmp
    return U_upd

def CMean(U, v):
    v_upd = v_update(U)
    U_upd = U_update(U,v_upd)
    for i in range(C):
        for k in range(N):
            if abs(U_upd[i][k] - U[i][k]) > e:
                U_upd, v_upd = CMean(U_upd, v_upd)
        
    return U_upd, v_upd

U_upd, v_upd = CMean(U_init, v_init)

plt.rcParams['font.size'] = 15
plt.scatter(X, U_upd[0])  
plt.scatter(X, U_upd[1]) 
plt.scatter(X, U_upd[2]) 
plt.scatter(X, U_upd[3]) 

plt.title("m = 2.10")
plt.xlabel("data value")
plt.ylabel("membership value")
plt.show()