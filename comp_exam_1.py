#Name: Swati Sriram Mani

import numpy as np
import matplotlib.pyplot as plt

N=500
samplepoints=10
x=np.linspace(0,1,samplepoints)
#y=np.sin(2*np.pi*x).reshape(1,-1)
noise=0.2+np.random.random()
t=np.sin(2*np.pi*x)
for i in range(0,9):
    t[i]=t[i]+((-1)**i)*noise   #y

xsin=np.linspace(0,1,N)
ysin=np.sin(2*np.pi*xsin)

plt.plot(x,t,'bo',xsin,ysin,'g-')
plt.ylim(-4,4)
plt.show()

def y(x, W, M):
    Y = np.array([W[i] * (x ** i) for i in range(M + 1)])
    return Y.sum()


# (1.2),(1.122),(1.123)
def E(x, t, M):
    A = np.zeros((M + 1, M + 1))
    for i in range(M + 1):
        for j in range(M + 1):
            A[i, j] = (x ** (i + j)).sum()

    T = np.array([((x ** i) * t).sum() for i in range(M + 1)])
    return np.linalg.solve(A, T)




for M in [0, 1, 3, 9]:
    W = E(x,t,M)
    #print(W)
    y_estimate = [y(i, W, M) for i in x]
    '''plt.plot(x, y_estimate, 'r-')
    plt.plot(xsin, ysin, 'g-')
    plt.plot(x, t, 'bo')
    plt.show()'''

#Regularization for M=9 using different values for Lambda



M = 9
lam = [0,(10**(-20)),(10**(-10)),(10**(-6)),(10**(-4)),(10**(-2)),1]

def phi(x, M):
    return x[:,None] ** np.arange(M + 1)


for i in range(len(lam)):
    phi_x = phi(x, M)
    S_0 = phi_x.T.dot(phi_x) + lam[i] * np.eye(M+1)
    y_0 = t.dot(phi_x)

    coeff = np.linalg.solve(S_0, y_0)[::-1]
    print("Weights: ",coeff)
    f = np.poly1d(coeff)

    xx = np.linspace(0, 1, N)

    fig, ax = plt.subplots()
    ax.plot(x, t, 'bo')
    ax.plot(xx, np.sin(2 * np.pi * xx), 'g-')
    ax.plot(xx, f(xx), 'r-')
    plt.ylim((-4, 4))
    plt.show()


