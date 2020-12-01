import numpy as np
import matplotlib.pyplot as plt
from MLbased import Opt,f
from BWAlgo import BWcal


state_0=np.array([1,0])
SigmaX=np.array([[0,1],[1,0]])
SigmaY=np.array([[0,-1j],[1j,0]])
SigmaZ=np.array([[1,0],[0,-1]])
H=(1.0/np.sqrt(2))*np.array([[1,1],[1,-1]])
rho_0=np.outer(state_0,state_0)


def HarrRandom(dim=2):
    # np.random.seed(seed)
    X = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    s = np.linalg.det(X)
    while s==0:
        X = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        s = np.linalg.det(X)
    U,_=np.linalg.qr(X)
    return U

def CaylayPath(X,Y,theta,dim=2):
    W=Y.dot(X.conj().T)
    eigh,eigvec=np.linalg.eig(W)
    edim=len(eigh)
    Ut=np.zeros((dim,dim))
    for i in range(edim):
        eigp=(1.0+eigh[i]-theta*(1.0-eigh[i]))/(1.0+eigh[i]+theta*(1.0-eigh[i]))
        Ut=Ut+eigp*np.outer(eigvec[i],eigvec[i].conj())

    return Ut.dot(X)

def Depolarize(rho,eps,dim=2):
    return eps*np.eye(dim)/dim+(1-eps)*rho


def AccSample(U):
    p=state_0.dot(U).dot(state_0)
    return p*p.conj()

def ErrSample(U,eps):
    x=AccSample(U)
    x=x+eps*np.random.normal()
    return x

U_f=HarrRandom()

Theta=[]
Y=[]
Err_Y=[]
for i in range(20):
    theta=1-0.01*i
    y=AccSample(CaylayPath(SigmaZ,U_f,theta))
    err_y=ErrSample(CaylayPath(SigmaZ,U_f,theta),0.01)
    Theta.append(theta)
    Y.append(y)
    Err_Y.append(err_y)
Theta=np.array(Theta)
Y=np.array(Y)
Err_Y=np.array(Err_Y)
plt.plot(Theta,Y,color='blue',linewidth=2.5)
# print(Y)
# a,b=Opt(Theta,Y,0.01,20000)
# reg,_,_=f(Theta,a,b)
reg=BWcal(Theta,Y)
plt.plot(Theta,reg,color='red')
plt.show()













