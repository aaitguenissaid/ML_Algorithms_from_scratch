import numpy as np
import matplotlib.pyplot as plt
    
class Gradient_descent_3d(object):
    def __init__(self, inf, sup, learning_rate, max_iters, f):
        self.lr = learning_rate
        self.max_iters = max_iters
        self.inf = inf
        self.sup = sup 
        self.f = f
        self.w1data = []
        self.w2data = []
        self.Ldata = []
                        
    def compute_gradient(self, w1,w2,f):
        if(f=='p'):
            return 2*w1 + w2, 2*w2 + w1
        
        elif(f=='c'):
            return w1/(np.sqrt(w1**2 + w2**2)), w2/(np.sqrt(w1**2 + w2**2))
        
        elif(f=='l'):
            ew=np.exp(-w1-w2)
            return (-ew/(1+ew)), (-ew/(1+ew)) 
        
        elif(f=='e'):
            return -np.exp(-w1-w2), -np.exp(-w1-w2) 
        
        elif(f=='b'):
            V = 3.0+w1**2+w2**2
            ww = (w1**2+w2**2)/4
            return (-6*w1*V*np.sin(ww)-24*w1*np.cos(ww))/V**2 , -(6*w2*V*np.sin(ww)-24*w2*np.cos(ww))/V**2
        
        elif(f=='r'):
            a=0.5
            b=10
            return -2 * (a-w1)-(4 * b * w1 * (w2 - w1**2)) , 2 * b * (w2 - w1**2)
        
        elif(f=='bu'):
            return -np.sin(5 * w2) * np.sin(5 * w1), np.cos(5 * w1) * np.cos(5 * w2)        
    
    def fit(self):
        self.w1data.append(self.inf)
        self.w2data.append(self.sup)
        self.Ldata.append(Functions(self.w1data[0], self.w2data[0], self.f))

        for epoch in range(1, self.max_iters):
            gradw1, gradw2 = self.compute_gradient(self.w1data[epoch-1], self.w2data[epoch-1], self.f)
            w1 = self.w1data[epoch - 1] - self.lr * gradw1
            w2 = self.w2data[epoch - 1] - self.lr * gradw2
            L = Functions(w1, w2, self.f)
            self.w1data.append(w1)
            self.w2data.append(w2)
            self.Ldata.append(L)


def Functions(w1,w2,f):
    if(f=='p'):
        return w1**2 + w1 * w2 + w2**2
        
    elif(f=='c'):
        return np.sqrt(w1**2 + w2**2)
    
    elif(f=='l'):
        return np.log(1 + np.exp(-w1 - w2)) # +0.2*(x**2+y**2)
    
    elif(f=='e'):
        return np.exp(-w1 - w2)
    
    elif(f=='b'):
        return 12 * np.cos((w1**2+w2**2)/4)/(3.0+w1**2+w2**2)
    
    elif(f=='r'):
        a=0.5
        b=10
        return (a-w1)**2+b*(w2-w1**2)**2
    
    elif(f=='bu'):
        return np.cos(5*w1) * np.sin(5*w2)/5

def PlotFunc3D(inf, sup, title, f, w1data, w2data, Ldata):
    fig = plt.figure()
    ax = plt.axes(projection='3d', computed_zorder=False)
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('L')
    ax.set_title(title)
    
    x = np.linspace(inf, sup, 20)
    y = np.linspace(inf, sup, 20)
    X, Y = np.meshgrid(x, y)
    Z = Functions(X, Y, f)
    ax.plot_surface(X, Y, Z, cmap='viridis', zorder=0)
    ax.plot3D(w1data, w2data, Ldata, '.', color="red");
    ax.view_init(35, 110)
