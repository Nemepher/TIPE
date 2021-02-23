import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sc


def f (x) : 
    if x>=0 : return 1 
    else : return 0
f=np.vectorize(f)


def h (x,b) : 
    if x>=0 and x<=b : return 1 
    else : return 0
h=np.vectorize(h)

g = lambda x, sig : 1/(np.sqrt(2*np.pi)*sig)*np.exp(-(x**2)/(2*sig))
gpp = lambda x,sig : 1/(np.sqrt(2*np.pi)*sig**2)*(1 - x**2/(sig**2))*np.exp(-(x**2)/(2*sig))


X = np.linspace(-7,7,200)
plt.plot(0.4*f(X))
plt.plot(-g(X,1))
plt.axis("off")
plt.show()


X = np.linspace(-5,10,400)
plt.plot(X,0.2*h(X,5))
plt.plot(X,-sc.convolve(h(X,5),gpp(X,1), mode = "same")/sum(h(X,5)))
plt.axis("off")
plt.show()



X = np.linspace(-10,15,400)
plt.plot(sc.convolve(gpp(X,1),h(X,10), mode = "same") )
plt.plot(sc.convolve(gpp(X,1),h(X,5), mode = "same") )
plt.plot(sc.convolve(gpp(X,1),h(X,3), mode = "same") )
plt.axis("off")
plt.show()

