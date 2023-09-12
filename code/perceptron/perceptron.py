def unit_step_function(x):
    return 1 if (x > 0) else -1

class Perceptron(object):
    def __init__(self, learning_rate, max_iters):
        self.lr = learning_rate
        self.max_iters = max_iters
        self.activation_func = unit_step_function
        self.weights = None
        self.bias = None
        
    def predict_single(self, inputs):
        res = self.bias + sum([w * i for w, i in zip(self.weights, inputs)])
        return self.activation_func(res)
        
    def predict(self, x):
        return [self.predict_single(x[i]) for i in range(len(x))]        
        
    def fit(self, x, y):
        n_samples = len(x)
        n_features = len(x[0])
        
        self.bias = 0.0
        self.weights = [0.0 for i in range(n_features)]
        epoch = 0
        error = 1
        
        while(epoch < self.max_iters and error != 0): # epochs loop
            error = 0
            for i in range(n_samples): # loop through all the data
                if((y[i] != self.predict_single(x[i]))): 
                    error += 1
                    self.bias += self.lr * y[i]
                    for j in range(n_features):
                        self.weights[j] += self.lr * y[i] * x[i][j]
            epoch=epoch+1