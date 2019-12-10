import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data_funct_rule.csv")

C_arr = [2, 3, 5, 7]
C = C_arr[0]
N = len(df)
U_init = np.random.rand(C, N)

for j in range(N):
    temp = 0
    for i in range(C):
        temp += U_init[i][j]
    for i in range(C):
        U_init[i][j] = U_init[i][j]/temp

X = np.array(df.X)
Y = np.array(df.Y)

m = 2.1
e = 0.01

v_init = np.zeros(C)
w_init = np.zeros(C)

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

def w_update(U):
    w = np.zeros(C)
    for i in range(C):
        tmp = 0
        t = 0
        for k in range(N):
            t += U[i][k]**m * Y[k]
        for k in range(N):
            tmp += U[i][k]**m
        w[i] = t/tmp
    return w

def U_update(U, v, w):
    U_upd = np.matrix.copy(U)
    for i in range(C):
        for k in range(N):
            tmp = 0
            for j in range(C):
                tmp += ((((X[k] - v[i])**2 + (Y[k] - w[i])**2)**1/2)/(((X[k] - v[j])**2 + (Y[k] - w[j])**2)**1/2))**(1/(m-1))
            U_upd[i][k] = 1/tmp
    return U_upd

def CMean(U, v, w):
    v_upd = v_update(U)
    w_upd = w_update(U)
    U_upd = U_update(U,v_upd, w_upd)
    for i in range(C):
        for k in range(N):
            if abs(U_upd[i][k] - U[i][k]) > e:
                U_upd, v_upd, w_upd = CMean(U_upd, v_upd, w_upd)
        
    return U_upd, v_upd, w_upd

U, v, w = CMean(U_init, v_init, w_init)

plt.rcParams['font.size'] = 15

plt.scatter(X, U[0])  
plt.scatter(X, U[1]) 
#plt.scatter(X, U[2]) 
#plt.scatter(X, U[3]) 
#plt.scatter(X, U[4]) 
#plt.scatter(X, U[5])
#plt.scatter(X, U[6])

plt.title("m = 2.10")
plt.xlabel("X")
plt.ylabel("membership value")
plt.show()

plt.scatter(U[0], Y)  
plt.scatter(U[1], Y) 
#plt.scatter(U[2], Y) 
#plt.scatter(U[3], Y)
#plt.scatter(U[4], Y)
#plt.scatter(U[5], Y)
#plt.scatter(U[6], Y)

plt.title("m = 2.10")
plt.xlabel("membership value")
plt.ylabel("Y")
plt.show()

U_init = np.random.rand(C, N)
v_init = np.zeros(C)

for j in range(N):
    temp = 0
    for i in range(C):
        temp += U_init[i][j]
    for i in range(C):
        U_init[i][j] = U_init[i][j]/temp
    
def U_x_update(U, v):
    U_upd = np.matrix.copy(U)
    for i in range(C):
        for k in range(N):
            tmp = 0
            for j in range(C):
                tmp += (abs(X[k] - v[i])/abs(X[k] - v[j]))**(2/(m-1))
            U_upd[i][k] = 1/tmp
    return U_upd

def CMean_x(U, v):
    v_upd = v_update(U)
    U_upd = U_x_update(U,v_upd)
    for i in range(C):
        for k in range(N):
            if abs(U_upd[i][k] - U[i][k]) > e:
                U_upd, v_upd = CMean_x(U_upd, v_upd)
        
    return U_upd, v_upd

U_x_upd, v_x_upd = CMean_x(U_init, v_init)

coef = []
x_clusters = []
y_clusters = []
u_clusters = []
for i in range(C):
    x = []
    y = []
    u = []
    for k in range(N):
        if U_x_upd[i][k] >= 0.5:
            x.append(X[k])
            y.append(Y[k])
            u.append(U_x_upd[i][k])
    x_clusters.append(x)
    y_clusters.append(y)
    u_clusters.append(u)
    
    const = 0
    a0 = 0
    a1 = 0
    for j in range(len(x)):
        const += u[j]**2 + (u[j]*x[j])**2
        #const += (u[j]*x[j])**2
        a0 += y[j] * u[j]
        a1 += y[j] * u[j] * x[j]
    a0 = a0 / const
    a1 = a1 / const
    
    coef.append([a0, a1])

y_pred = []
for i in range(C):
    y = []
    for j in range(len(x_clusters[i])):
        y.append(u_clusters[i][j]*(coef[i][0] + x_clusters[i][j]*coef[i][1]))
    y_pred.append(y)

plt.plot(x_clusters[0], y_pred[0])
plt.plot(x_clusters[1], y_pred[1])
#plt.plot(x_clusters[2], y_pred[2])
#plt.plot(x_clusters[3], y_pred[3])
#plt.plot(x_clusters[4], y_pred[4])
#plt.plot(x_clusters[5], y_pred[5])
#plt.plot(x_clusters[6], y_pred[6])
plt.scatter(df.X, df.Y, c='grey')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

def MSE():
    cost = 0
    for i in range(C):
        for j in range(len(x_clusters[i])):
            cost += (Y[i] - y_pred[i][j])**2
    cost = cost/N
    return cost