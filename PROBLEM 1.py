import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt

#%%
data = np.loadtxt('C:/Users/Kunal Prasad/OneDrive/Desktop/phd 2024/University of Amsterdam/Problem/WS2_monolayer_nk.txt',dtype='float')
X = data[:,0]
nr = data[:,1]
ni = data[:,2]
#%%

plt.plot(X,nr)
plt.plot(X,ni)
plt.legend(["Re(n)","Im(n)"])
plt.xlabel("Wavelenght(nm)")
plt.ylabel("Refractive Index")
plt.grid(True)
plt.title("Refractive Index of WS2 monolayer")
plt.savefig("Refractive Index of WS2 monolayer.png", format='png',dpi=300)
plt.show()
#%%
N=1000
n=nr+1j*ni
d=0.618
k=2*np.pi/X
#%%
r1=(1-n)/(n+1)
t1=(2)/(n+1)
t2=(2*n)/(n+1)
r2=(n-1)/(n+1)
r=np.zeros(len(X),dtype='complex')

#%%
for i in range(len(X)):
    S=0.0+0j
    for m in range(N):
        S=S+(r2[i]**(2*m+1))*(np.exp(2*1j*d*k[i]*n[i]*(m+1)))
    r[i]=r1[i]+t1[i]*t2[i]*S
#%%
tt= (np.exp(2j*k*n*d)*t1*t2)/(1-r2*r2*np.exp(2j*k*n*d))
#%%
T= abs(tt)**2
 
#%%
R = r*np.conjugate(r)
plt.plot(X,R)
plt.legend(["Total Reflectance"])
plt.xlabel("Wavelenght(nm)")
plt.ylabel("Reflectance")
plt.grid(True)
plt.title("Reflectance of WS2 monolayer at 0.618nm thickness")
plt.savefig("Reflectance of WS2 monolayer at 0.618nm thickness", format='png',dpi=300)
plt.show()

#%%
A=1-R-T
plt.plot(X,T)
plt.legend(["Total Transmittance"])
plt.xlabel("Wavelenght(nm)")
plt.ylabel("Transmittance")
plt.grid(True)
plt.title("Transmittance of WS2 monolayer at 0.618nm thickness")
plt.savefig("Transmittance of WS2 monolayer at 0.618nm thickness", format='png',dpi=300)
plt.show()
#%%
plt.plot(X,A)
plt.legend(["Total Absorptance"])
plt.xlabel("Wavelenght(nm)")
plt.ylabel("Absorptance")
plt.grid(True)
plt.title("Absorptance of WS2 monolayer at 0.618nm thickness")
plt.savefig("Absorptance of WS2 monolayer at 0.618nm thickness", format='png',dpi=300)
plt.show()

#%%
plt.plot(X,np.real(r))
plt.plot(X,np.imag(r))
plt.legend(["Re(r)","Im(r)"])
plt.xlabel("Wavelenght(nm)")
plt.ylabel("Reflection Coefficient")
plt.grid(True)
plt.title("Reflection coeffiecient of WS2 monolayer at 0.618nm thickness")
plt.savefig("Reflectance of WS2 monolayer at 0.618nm thickness2", format='png',dpi=300)
plt.show()

#%%
#%%
plt.scatter(np.real(r),np.imag(r))
plt.ylabel("Im(r)")
plt.xlabel("Re(r)")
plt.grid(True)
plt.title("Reflectance of WS2 monolayer at 0.618nm thickness")
plt.savefig("Reflectance of WS2 monolayer at 0.618nm thickness3", format='png',dpi=300)
plt.show()
#%%
rr=np.zeros(len(X),dtype='complex')
dd=np.linspace(0.618,6.18,10)
maxR = np.zeros(len(dd))
maxX = np.zeros(len(dd))
for l in range(len(dd)):
    for i in range(len(X)):
        S=0.0+0j
        for m in range(N):
            S=S+r2[i]**(2*m+1)*(np.exp(2*1j*dd[l]*k[i]*n[i]*(m+1)))
        rr[i]=r1[i]+t1[i]*t2[i]*S
    rrr=(np.abs(rr))**2
    maxR[l]=np.max(rrr[93:150])
    maxX[l]=X[np.where(rrr[93:150]==rrr[93:150].max())]
    plt.plot(X,rr*np.conjugate(rr))
    plt.legend(["Total Reflectance"])
    plt.xlabel("Wavelenght(nm)")
    plt.ylabel("Reflectance")
    plt.grid(True)
    plt.title("Reflectance of WS2 monolayer at different thicknesses")

    
#%%
plt.scatter(dd,maxR)
plt.xlabel("Thickness(nm)")
plt.ylabel("Reflectance")
plt.grid(True)
plt.title("Reflection peak variation with WS2 thickness")
plt.show()

#%%
plt.scatter(dd,maxX)
plt.xlabel("Thickness(nm)")
plt.ylabel("Reflectance peak wavelength(nm)")
plt.grid(True)
plt.title("Reflection peak wavelength variation with WS2 thickness")
plt.show()

#%%
E = 1240/X
E = np.flip(E)
#%%
Chi = n*n - 1
Chi = np.flip(Chi)
#%%
plt.plot(E,np.real(Chi))
plt.plot(E,np.imag(Chi))
plt.legend(["Re(Chi)","Im(Chi)"])
plt.xlabel("Photon Energy(eV)")
plt.ylabel("Susceptibility")
plt.grid(True)
plt.title("Susceptibility of WS2 monolayer")
plt.savefig("Susceptibility Index of WS2 monolayer", format='png',dpi=300)
plt.show()
#%%
def XchiR(E,Er,Enr,X0):
    return X0+np.real(1240*Er/(2*np.pi*0.618*2.01462*((E-2.01462)+0.5j*Enr)))

def XchiI(E,Er,Enr):
    return np.imag(1240*Er/(2*np.pi*0.618*2.01462*((E-2.01462)+0.5j*Enr)))

#%%
indexi=95
indexf=130
ydataR = np.real(Chi[indexi:indexf])
ydataI = np.imag(Chi[indexi:indexf])
xdata = E[indexi:indexf]


#%%
from scipy.optimize import curve_fit
#%%
popt, pcov = curve_fit(XchiI, xdata, ydataI)

#%%
#%%
plt.plot(E,np.imag(Chi))
plt.plot(E,XchiI(E,popt[0],popt[1]))
#plt.legend(["Re(Chi)","Im(Chi)"])
plt.xlabel("Photon Energy(eV)")
plt.ylabel("Im(Susceptibility)")
plt.grid(True)
plt.title("Optimization of lorentzian susceptibility")
plt.savefig("Optimization of lorentzian susceptibilityI.png", format='png',dpi=300)
plt.show()

#%%
plt.plot(E,Chi)
plt.plot(E,XchiR(E,popt[0],popt[1],20.5))
#plt.legend(["Re(Chi)","Im(Chi)"])
plt.xlabel("Photon Energy(eV)")
plt.ylabel("Re(Susceptibility)")
plt.grid(True)
plt.title("Optimization of lorentzian susceptibility")
plt.savefig("Optimization of lorentzian susceptibilityR.png", format='png',dpi=300)
plt.show()

#%%
Er=abs(popt[0])*1000
Enr=abs(popt[1])*1000
Q=Er/(Er+Enr)
print('Non - radiative decay rate = ',Enr,"meV")
print('Radiative decay rate = ',Er,"meV")
print('Quantum Efficiency = ',Q)


#%%
plt.scatter(np.real(Chi),np.imag(Chi))
plt.ylabel("Im(Susceptibility)")
plt.xlabel("Re(Susceptibility)")
plt.grid(True)
plt.title("Susceptibility of WS2 monolayer")
plt.savefig("Susceptibility Index of WS2 monolayer2", format='png',dpi=300)
plt.show()


#%%
