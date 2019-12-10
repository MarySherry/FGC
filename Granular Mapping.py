import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

df = pd.read_csv("data_granular_mapping.csv")

e1 = [e/10 for e in range(2, 11, 1)]
e2 = [1.2 - e for e in e1]

tmp = 0
t = 0
for i in range(len(df)):
    tmp += (df.X[i] - df.X.mean()) * (df.Y[i] - df.Y.mean())
    t += (df.X[i] - df.X.mean()) * (df.X[i] - df.X.mean())
a1 = tmp/t
a0 = df.Y.mean() - a1 * df.X.mean()

def gran_param(e_1, e_2):
    A0 = [a0 * (1 - e_1), a0 * (1 + e_1)]
    A1 = [a1 * (1 - e_2), a1 * (1 + e_2)]
    return A0, A1

def gran_output(A_0, A_1):
    Y = []
    for i in range(len(df)):
        Y.append([A_0[0] + df.X[i]*A_1[0], A_0[1] + df.X[i] * A_1[1]])
    return Y

y_max = df.Y.max()
y_min = df.Y.min()
R = y_max - y_min

def Q_func(E1, E2):
    Q = []
    for e in range(len(E1)):
        A0, A1 = gran_param(E1[e], E2[e])
        Y = gran_output(A0, A1)
        cov = 0
        sp = 0
        for i in range(len(df)):
            if df.Y[i] <= Y[i][1] and df.Y[i] >= Y[i][0]:
                cov +=1
            if (Y[i][1] - Y[i][0])/R < 1:
                sp += (1 - (Y[i][1] - Y[i][0])/R)
        Q.append(1/len(df) * cov * 1/len(df) * sp)

    return np.array(Q)

fig = plt.figure()
ax = plt.axes(projection='3d')

Q = Q_func(e1, e2)
ax.plot3D(e1, e2, Q)
ax.set_xlabel('e1')
ax.set_ylabel('e2')
ax.set_zlabel('Q')
ax.view_init(60, 35)
plt.show()

plt.scatter(df.X, df.Y)
plt.xlabel("X")
plt.ylabel("Y")

A0, A1 = gran_param(e1[8], e2[8])
Y = gran_output(A0, A1)
for i in range(len(df)):
    plt.plot([df.X[i],df.X[i]], Y[i])
plt.show()