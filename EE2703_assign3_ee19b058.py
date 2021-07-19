"""   ASSIGNMENT 3 
	FITTING DATA TO MODELS
NAME: VINAYAKA SRIDHAR K V V
ROLL NO.: EE19B058
"""
filename = 'fitting.dat'	#defining filename as the required file fitting.dat
#importing the required modules
import numpy as np
from pylab import *
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib import cm
#loading the text file and taking the column 0 as time
time=np.loadtxt(filename,usecols=0)
#taking column 1 input 
f=np.loadtxt(filename,usecols=1)
#defining a function Fk(k) which returns the element k of the first column of data
def Fk(k):
	fk=np.loadtxt(filename,usecols=1)[k]
	return fk

J2=np.loadtxt(filename,usecols=1)

#creating a vector J2 consisting of the bessel function values at time values in the time array 
for T in range(len(time)):
	J2[T]=sp.jn(2,0.1*T)

#creating a matrix M made from column vectors J2 and time 
M=c_[J2,time]

#defining a function g(t,A,B) that computes the value of function for different coefficients A,B and at an instant t
def g(t,A,B):
	coeff=np.array([[A],[B]])
	gmat=np.dot(M,coeff)
	gt=gmat[int(10*t)]
	return gt
#creating the vector containing values of exact function in the time interval 0 to 100
A=1.05 # coefficients of exact function
B=-0.105
coeff=np.array([[A],[B]]) #The coefficient vector
gexact=np.dot(M,coeff) #the exact function vector

#calculating the error for function with different noise levels with respect to exact function
Eij=np.zeros([21,21])
for a in np.arange(0,21,1):
	A=a/10
	for b in np.arange(-20,1,1):	
		B=b/100
		for k in range(101):
			Eij[a][b]=Eij[a][b] + (1/100)*((Fk(k)-g(k/10,A,B))**2)

# Now plotting the required plots

t=np.linspace(0,10,101) #time interval definition 
fig0=plt.figure()
#assigning each column of the data to different variables 
#each column in the data corresponds to function values with different noise levels
f1=np.loadtxt(filename,usecols=1)
f2=np.loadtxt(filename,usecols=2)
f3=np.loadtxt(filename,usecols=3)
f4=np.loadtxt(filename,usecols=4)
f5=np.loadtxt(filename,usecols=5)
f6=np.loadtxt(filename,usecols=6)
f7=np.loadtxt(filename,usecols=7)
f8=np.loadtxt(filename,usecols=8)
f9=np.loadtxt(filename,usecols=9)
#plotting all the functions with different noise values with a standard deviation sigma
plt.plot(t,f1,color='blue',label='sigma=0.1')
plt.plot(t,f2,color='red',label='sigma=0.056')
plt.plot(t,f3,color='orange',label='sigma=0.032')
plt.plot(t,f4,color='olive',label='sigma=0.018')
plt.plot(t,f5,color='green',label='sigma=0.01')
plt.plot(t,f6,color='purple',label='sigma=0.006')
plt.plot(t,f7,color='pink',label='sigma=0.003')
plt.plot(t,f8,color='magenta',label='sigma=0.002')
plt.plot(t,f9,color='cyan',label='sigma=0.001')
plt.plot(t,gexact,color='black',label='exact curve')
legend()
#labelling the x and y axes correspondingly
xlabel(r't $\rightarrow$')
ylabel(r'f(t) + noise $\rightarrow$')
title('Q4.Data to be fitted to theory')	#Giving the plot a title

#copying values of the first column of the data to a array named data for convenience
data=[0.0]*101
for d in range(101):
	data[d]=f[d]
	
sigma=np.logspace(-1,-3,9)	#array of the standard deviations of the noise function
sigma1=sigma[0]

print(sigma) #printing the standard deviation array

#plotting the errorbars along with the exact function 
fig1=plt.figure()
plt.plot(t,gexact,color='black',label='exact curve')#exact curve plot in black color
errorbar(t[::5],data[::5],sigma1,fmt='ro',markersize='5',label='Errorbar') #errorbar plot
legend()
#labelling the axis
xlabel(r't $\rightarrow$')
title('Q5.Data points for sigma for first column of data') #title assignment

#The CONTOUR PLOT of the error 
A=np.linspace(0,2,21)
B=np.linspace(-0.2,0,21)
A,B=np.meshgrid(A,B)
z=Eij
fig2=plt.figure()
ax=fig2.add_subplot(111)
C=ax.contour(A,B,z,levels=20)
clabel(C,[0,0.025,0.05,0.1])	#labelling the contour lines with their corresponding values
scatter(1.05,-0.105,color='red',label='exact location')	#plotting the exact location of the actual coefficients
legend()
xlabel(r'A $\rightarrow$')
ylabel(r'B $\rightarrow$')
title('Q8.Contour plot of Epsilon_ij')

#computing the estimated coefficients for each column using least squares method
est_coeff1=np.linalg.lstsq(M,f1,rcond=None)
est_coeff2=np.linalg.lstsq(M,f2,rcond=None)
est_coeff3=np.linalg.lstsq(M,f3,rcond=None)
est_coeff4=np.linalg.lstsq(M,f4,rcond=None)
est_coeff5=np.linalg.lstsq(M,f5,rcond=None)
est_coeff6=np.linalg.lstsq(M,f6,rcond=None)
est_coeff7=np.linalg.lstsq(M,f7,rcond=None)
est_coeff8=np.linalg.lstsq(M,f8,rcond=None)
est_coeff9=np.linalg.lstsq(M,f9,rcond=None)

#the Aerr that is the error in coefficient A from the actual value 1.05
Aerr=[abs(est_coeff1[0][0]-1.05),abs(est_coeff2[0][0]-1.05),abs(est_coeff3[0][0]-1.05),abs(est_coeff4[0][0]-1.05),abs(est_coeff5[0][0]-1.05),abs(est_coeff6[0][0]-1.05),abs(est_coeff7[0][0]-1.05),abs(est_coeff8[0][0]-1.05),abs(est_coeff9[0][0]-1.05)]

#Berr is the error in coefficient B from the actual value -0.105
Berr=[abs(est_coeff1[0][1]+0.105),abs(est_coeff2[0][1]+0.105),abs(est_coeff3[0][1]+0.105),abs(est_coeff4[0][1]+0.105),abs(est_coeff5[0][1]+0.105),abs(est_coeff6[0][1]+0.105),abs(est_coeff7[0][1]+0.105),abs(est_coeff8[0][1]+0.105),abs(est_coeff9[0][1]+0.105)]

#plotting the variation of error in coefficients with noise
fig3=plt.figure()
scatter(sigma,Aerr,color='red',label='Aerr')		#error in A plotted in red dots
scatter(sigma,Berr,color='green',label='Berr')	#error in B plotted in green dots	
legend()
#labelling the axes
xlabel(r'sigma $\rightarrow$')
ylabel(r'MS error $\rightarrow$')
title('Q10.Variation of error with noise')		#Title assignment

#plotting the variation in 
fig4=plt.figure()
plt.loglog(sigma,Aerr,label='Aerr')
plt.loglog(sigma,Berr,label='Berr')
legend()
xlabel(r'sigma $\rightarrow$')
ylabel(r'MS error $\rightarrow$')
title('Q11.Variation of error with noise')

plt.show()



