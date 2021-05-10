import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sc


def f (x) : 
    if x>=0 : return 1 
    else : return 0
f=np.vectorize(f)


def h (x,b) : 
    if x>=-b and x<=b : return 1 
    else : return 0
h=np.vectorize(h)

g = lambda x, sig : 1/(np.sqrt(2*np.pi)*sig)*np.exp(-(x**2)/(2*sig))
gpp = lambda x,sig : 1/(np.sqrt(2*np.pi)*sig**2)*(1 - x**2/(sig**2))*np.exp(-(x**2)/(2*sig))

"""
X = np.linspace(-7,7,200)
plt.plot(0.4*f(X),linewidth=4.0)
plt.plot(-g(X,1),linewidth=4.0)
plt.axis("off")
plt.show()


X = np.linspace(-6,6,200)
h2=h(X,1.5)
plt.plot(X,0.1*h2,linewidth=4.0)
plt.plot(X,-sc.convolve(h2,gpp(X,1), mode = "same")/sum(h2), label="σ=1",linewidth=4.0)
plt.plot(X,-sc.convolve(h2,gpp(X,2), mode = "same")/sum(h2), label="σ=2",linewidth=4.0)
plt.plot(X,-sc.convolve(h2,gpp(X,3), mode = "same")/sum(h2), label="σ=3",linewidth=4.0)
plt.axis("off")
plt.show()


X = np.linspace(-9,9,200)
l=4
fig = plt.figure()
for k in range(l):
    ax=fig.add_subplot(1,l,k+1)
    h2 = h(X,5-k*1.4)
    ax.plot(X,0.1*h2, label="largeur: "+str(5-k*1.5),linewidth=4.0)
    ax.plot(X,-sc.convolve(h2,gpp(X,1), mode = "same")/sum(h2),linewidth=4.0)
    ax.axis("off")
    ax.set_ylim(-0.2,0.2)

plt.show()

"""