import numpy as np

def f(theta,a,b):
    vec=np.vander(theta,N=5,increasing=True).T
    p=a.dot(vec)
    q=b.dot(vec)
    return p/q,p,q

def grad(theta,a,b):
    fx,p,q=f(theta,a,b)
    vec=np.vander(theta,N=5,increasing=True).T
    #print(vec.shape)
    grad_a=1/q*vec
    grad_b=-fx/q*vec
    return grad_a,grad_b


def Opt(Theta,Y,eta,epoch):
    a=np.random.randn(5)
    b=np.random.randn(5)*np.array([1,0,1,0,1])
    N=Theta.shape[0]
    loss=[]
    for i in range(epoch):
        fx, _, _ = f(Theta, a, b)
        l = 1 / N * (fx - Y).dot(fx - Y)
        grad_a, grad_b = grad(Theta, a, b)
        a = a - 2 * eta / N * (fx - Y).dot(grad_a.T)
        b = b - 2 * eta / N * (fx - Y).dot(grad_b.T)
        b = b * np.array([1, 0, 1, 0, 1])
        loss.append(l)
        if i%1000==0:
            print("loss=",l)
    return a,b


# Theta=np.array([0.8,1,1.2])
# Y=np.array([0.75,1,0.8])
# print(Opt(Theta,Y,eta=0.01,epoch=1000))

# theta=np.array([0.5])
# a=np.array([1,2,1,4,5])
# b=np.array([1,0,3,0,1])
# a1=np.array([1,2,1,4,5])
# b1=np.array([1,0,3.0001,0,1])
# print(grad(theta,a,b))
# print((f(theta,a,b1)[0]-f(theta,a,b)[0])/0.0001)