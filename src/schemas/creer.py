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

def gauplot(centers, radiuses, xr=None, yr=None):
        nx, ny = 1000.,1000.
        xgrid, ygrid = np.mgrid[xr[0]:xr[1]:(xr[1]-xr[0])/nx,yr[0]:yr[1]:(yr[1]-yr[0])/ny]
        im = xgrid*0 + np.nan
        xs = np.array([np.nan])
        ys = np.array([np.nan])
        fis = np.concatenate((np.linspace(-np.pi,np.pi,100), [np.nan]) )
        cmap = plt.cm.gray
        cmap.set_bad('white')
        thresh = 3
        for curcen,currad in zip(centers,radiuses):
                curim=(((xgrid-curcen[0])**2+(ygrid-curcen[1])**2)**.5)/currad*thresh
                im[curim<thresh]=np.exp(-.5*curim**2)[curim<thresh]
                xs = np.append(xs, curcen[0] + currad * np.cos(fis))
                ys = np.append(ys, curcen[1] + currad * np.sin(fis))
        plt.imshow(im.T, cmap=cmap, extent=xr+yr)
        plt.plot(xs, ys, 'r-')

gauplot([(2,2)], [2], [-1,10], [-1,10])
plt.show()
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