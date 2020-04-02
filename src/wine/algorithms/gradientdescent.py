import numpy as np
import random as rand

class GradientDescent:
    def __init__(self, delta, learning_rate, data):
        '''
        Initialize hyperparameters 
        '''
        self.delta = delta
        self.lamda = 2
        self.learning_rate = learning_rate
        self.data = data

    def fit(self,optimizer, num_iterations=10, callbacks=[]):
        '''
        Run iterations and obtain loss after each iteration
        '''
        self.get_subsets()
        self.X = self.subsets[:, :, :11]
        self.y = self.subsets[:, :, 11]

        self.W = np.ones_like(self.X[1,1])
        print(self.X.shape, self.y.shape)
        for j in range(num_iterations):
            preds = self.predict(self.X)
            gradient = np.zeros(11)
            for subset_X, subset_y, subset_preds in zip(self.X, self.y, preds):
                correct = subset_y.argmax()
                score_correct = subset_preds[correct]
                for xi,predicted_score in zip(subset_X,subset_preds):
                    gradient -= (xi * max(0, predicted_score + self.delta - score_correct))
            
            gradient /= 100*10

            loss = self.calculate_loss(preds, self.y)
            print("Iteration:  %d, Loss: %.2f"%(j+ 1, loss))
            self.W = np.add(self.W, self.learning_rate * gradient - (2 * self.lamda * self.W))
            
    def predict(self, X):
        '''
        Return predictions based on weights calculated
        '''
        preds = []
        for subset in X:
            temp = []
            for x in subset:
                scores = np.sum(self.W.dot(x))
                temp.append(scores)
            preds.append(temp)
        return np.asarray(preds)

    def get_subsets(self):
        '''
        Split data into 100 subsets
        '''
        subsets = []
        for i in range(100):        
            subsets.append(rand.sample(self.data, 10))
        self.subsets = np.asarray(subsets)
        return

    def calculate_loss(self, preds, y):
        '''
        Calculate loss
        '''
        for pi, yi in zip(preds, y):
            correct = yi.argmax()
            score_correct = pi[correct]
            
            loss = 0
            for i, pred in enumerate(pi):
                loss += max(0, pred + self.delta - score_correct)            
                    
            return loss
