import numpy as np

def gcd(a,b):
    while b.dot(b.conj())>1e-10:
        t=b
        _,b=np.polydiv(a,b)
        a=t
    return a

def linearsolve(Theta,Y,k=4,t=2):
    M1=np.vander(Theta,N=1+k+t,increasing=True)
    M2=np.vander(Theta,N=k+t,increasing=True)
    M2=np.multiply(M2,Y[:,np.newaxis])
    M=np.concatenate((M1,-M2),axis=1)
    v=np.linalg.pinv(M).dot(Y*Theta**(k+t))
    a=v[:(k+t+1)]
    b=v[(k+t+1):]
    b=np.concatenate((b,np.array([1])))
    return a,b

def BWcal(Theta,Y,k=4,t=2):
    a,b=linearsolve(Theta,Y,k,t)
    a=a[::-1]
    b=b[::-1]
    #print(a,b)
    p=gcd(a,b)
    af,_=np.polydiv(a,p)
    bf,_=np.polydiv(b,p)
    af=af[::-1]
    bf=bf[::-1]
    print(af)
    print(bf)
    if len(af)>k+1 or len(bf)>k+1:
        print("Failed")
        return None
    else:
        vec=np.vander(Theta,N=1+k,increasing=True)
        lena=len(af)
        lenb=len(bf)

        af=np.concatenate((af,np.zeros(1+k-lena)))
        bf=np.concatenate((bf,np.zeros(1+k-lenb)))
        ans=af.dot(vec.T)/bf.dot(vec.T)
        return ans

# Theta=np.arange(5)
# Y=((1+2*Theta)/(1+Theta))
# Y=Y+0*1j
# print(BWcal(Theta,Y,k=1,t=1))

