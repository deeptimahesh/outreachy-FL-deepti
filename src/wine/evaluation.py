import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import random as rand

data = np.genfromtxt('data/winequality-white.csv',delimiter=';')[1:]
X = data[:,:11]
y = data[:,11]

def normalize(X):
    '''
    Normalize data
    '''
    X_flat = flatten(X)
    mu = X_flat.mean(axis=0)
    
    return [x - mu for x in X]

def get_subsets(data):
    '''
    Partition into subsets
    '''
    subsets = []
    for i in range(100):        
        subsets.append(rand.sample(list(data), 10))
    subsets = np.asarray(subsets)

    return subsets

# Detailed in the paper
def svm_loss(preds, ys, delta=0):
    '''
    Detailed in paper
    '''
    correct = ys.argmax()
    score_correct = preds[correct]
    
    loss = 0
    
    for i, pred in enumerate(preds):
        loss += max(0, pred + delta - score_correct)            
            
    return loss

def my_loss(preds, ys, delta = 0):
    correct = ys.argmax()
    score_correct = preds[correct]

    loss = 0
    for i, pred in enumerate(preds):
        loss += max(0, pred + delta - score_correct)
    return loss


X = normalize(X)
model = LogisticRegression(fit_intercept=False)
model.fit(X, y)

subsets = get_subsets(data)

model2 = LogisticRegression(fit_intercept=False,C=2, penalty='l2')
model2.fit(X, y)

# Calculate loss
loss = []
reg_loss = []
print("Computing losses...")
for subset in subsets:
    preds = model.predict(subset[:, :11])
    my_preds = model2.predict(subset[:, :11])

    score = np.argmax(preds)
    best_wine = subset[score][11]
    
    loss.append(svm_loss(preds, subset[:,11]))
    reg_loss.append(my_loss(my_preds, subset[:,11]))

# Visualize
plt.figure(figsize=(10,10))
plt.plot(loss, label="Given loss")
plt.plot(reg_loss, label="Regularized")
plt.legend()
plt.show()