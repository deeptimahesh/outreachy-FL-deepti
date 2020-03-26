# from gradient_descent import GradientDescent, AdaptiveGradientDescent, DecayedGradientDescent
# from rprop import RProp
# from adam import Adam

import numpy as np

class Adam:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.alpha = 0.01
        self.eps = 1e-8
        
        self.t = 0
        
        self.m = None
        self.v = None
        self.theta = None
    
    def __call__(self, gradient):
        if self.t == 0:
            self.m = np.zeros(gradient.shape)
            self.v = np.zeros(gradient.shape)
            self.theta = np.zeros(gradient.shape)
        
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        
        m_corrected = self.m / (1 - self.beta1**self.t)
        v_corrected = self.v / (1 - self.beta2**self.t)

        self.theta += self.alpha * m_corrected / (np.sqrt(v_corrected) + self.eps)
        
        return self.learning_rate * self.theta

class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
    def __call__(self, gradient):
        return self.learning_rate * gradient
    
class AdaptiveGradientDescent:
    def __init__(self, learning_rate, num_features):
        self.learning_rates = np.full(num_features, learning_rate)
        self.min = 0.0000001
        self.max = 50
        self.a = 1.2
        self.b = 0.5
        
        self.t = 0
        self.last_gradient = None
    
    def __call__(self, gradient):
        if self.t >= 1:
            for i in range(len(gradient)):
                if gradient[i] * self.last_gradient[i] > 0:
                    self.learning_rates[i] = min(self.learning_rates[i] * self.a, self.max)
                elif gradient[i] * self.last_gradient[i] < 0:
                    self.learning_rates[i] = max(self.learning_rates[i] * self.b, self.min)
            
        self.t += 1
        self.last_gradient = gradient
        
        return self.learning_rates * gradient
    
class DecayedGradientDescent:
    def __init__(self, learning_rate, decay):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epoch = 0
        
    def __call__(self, gradient):
        result = self.learning_rate * gradient
        
        self.learning_rate = self.learning_rate * 1 / (1 + self.decay * self.epoch)
        self.epoch += 1
            
        return result

class RProp:
    def __init__(self, learning_rate, num_features, min_value=0.0000001, max_value=50, alpha=1.2, beta=0.5):
        self.learning_rates = np.full(num_features, learning_rate)
        self.min = min_value
        self.max = max_value
        self.a = alpha
        self.b = beta
        
        self.t = 0
        self.last_gradient = None
    
    def __call__(self, gradient):
        if self.t >= 1:
            for i in range(len(gradient)):
                if gradient[i] * self.last_gradient[i] > 0:
                    self.learning_rates[i] = min(self.learning_rates[i] * self.a, self.max)
                elif gradient[i] * self.last_gradient[i] < 0:
                    self.learning_rates[i] = max(self.learning_rates[i] * self.b, self.min)
            
        self.learning_rates = np.round(self.learning_rates)
            
        self.t += 1
        self.last_gradient = gradient
        
        return self.learning_rates * np.sign(gradient)