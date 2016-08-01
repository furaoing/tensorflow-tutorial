import numpy as np
import pylab
import math
import random
from scipy.io import loadmat

mat_file_name = "ex5data1.mat"
mat_file_data = loadmat(mat_file_name)

dataX = mat_file_data.get("X")
datay = mat_file_data.get("y")

input_feature=2

t=list()
for i in range(input_feature):
        t.append(random.random()*100)

grad=input_feature*[None]

random_multiplier=1

x=list()
for item in dataX:
        x.append([1, item[0]])

y=list()
for item in datay:
    y.append(item)

dev=list()

def h(i):
	h=sum([t[k]*x[i][k] for k in range(input_feature)])
	return h

def gradient_descent(alpha, x, y, ep, max_iter,h):
    converged = False
    iter=0
    m=len(x) # number of samples
    n=len(t) # number of theta

    # total error, J(theta)
    J=0.5*sum([(h(i)-y[i])**2 for i in range(m)])
    
    # Iterate Loop
    while not converged:
        #for each training sample, compute the gradient (d/d_theta J(theta))
        for k in range(n):
            grad[k]=sum([(h(i)-y[i])*x[i][k] for i in range(m)])
        for k in range(n):
            t[k]-=alpha*grad[k]

        e = 0.5*sum([(h(i)-y[i])**2 for i in range(m)])
        print(e)
        
        dev.append(e) # obtain dev for plotting

        if abs(J-e) <= ep:
            converged= True

        J = e #update error
        iter +=1   # update iter

        if iter == max_iter:
            converged = True

    return t

gradient_descent(0.0001, x, y, -0.1, 50000000,h)


plot_y=[h(i) for i in range(len(dataX))]

plot_x=[x[0] for x in dataX]

pylab.plot(plot_x,plot_y)

pylab.plot(plot_x,y,'ro')

pylab.show()
print("Done!")
