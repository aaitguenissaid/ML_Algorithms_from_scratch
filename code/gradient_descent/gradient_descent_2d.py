import numpy as np
import matplotlib.pyplot as plt

class Gradient_descent(object):
    def __init__(self, start, learning_rate, max_iters, f):
        self.lr = learning_rate
        self.max_iters = max_iters
        self.start = start
        self.f = f
        self.wdata = []
        self.Ldata = []
                        
    def compute_gradient(self, w, f):
        if(f=='p'):
            return 2 * w

        elif(f=='c'):
            return 3 * w

        elif(f=='l'):
            return np.exp(w)

        elif(f=='e'):
            return -np.sin(w)

        elif(f=='b'):
            return 1/w

        elif(f=='r'):
            return 3 * w + 10 * w + 1
        
    def fit(self):
        self.wdata.append(self.start)
        self.Ldata.append(Functions(self.wdata[0], self.f))

        for epoch in range(1, self.max_iters):
            gradw = self.compute_gradient(self.wdata[epoch-1], self.f)
            w = self.wdata[epoch - 1] - self.lr * gradw
            L = Functions(w, self.f)
            self.wdata.append(w)
            self.Ldata.append(L) 

            
def Functions(w, f):
    if(f=='p'):
        return w ** 2
    
    elif(f=='c'):
        return w ** 3
    
    elif(f=='l'):
        return np.exp(w)
    
    elif(f=='e'):
        return np.cos(w)
    
    elif(f=='b'):
        return np.log(w)
    
    elif(f=='r'):
        return w**3 + 5*w**2 + w + 1
    
def PlotFunc2D(inf, sup, title, f, xdata, ydata):
    fig = plt.figure()
    X = np.arange(inf, sup, 0.1)
    Y = Functions(X, f)
    ax = plt.axes()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.plot(X, Y, zorder=0)
    ax.scatter(xdata, ydata, color='red', marker='.', zorder=1)
    