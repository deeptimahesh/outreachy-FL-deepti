import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

data = np.genfromtxt('winequality-white.csv',delimiter=';')[1:]
X = data[:,:11]
y = data[:,11]

def normalize(X):
    X_flat = flatten(X)
    mu = X_flat.mean(axis=0)
    return [x - mu for x in X]
def flatten(X):
    X_flat = []

    for x in X:
        X_flat += list(x)

    return np.array(X_flat)

from sklearn.linear_model import LogisticRegression
X = normalize(X)
model = LogisticRegression(fit_intercept=False)
model.fit(X, y)

import random as rand
subsets = []
for i in range(100):        
    subsets.append(rand.sample(list(data), 10))
subsets = np.asarray(subsets)
subsets.shape

# Detailed in the paper
def svm_loss(preds, ys, delta=0):
    correct = ys.argmax()
    score_correct = preds[correct]
    
    loss = 0
    
    for i, pred in enumerate(preds):
        loss += max(0, pred + delta - score_correct)            
            
    return loss

model2 = LogisticRegression(fit_intercept=False,C=2, penalty='l2')
model2.fit(X, y)

def my_loss(preds, ys, delta = 0):
    correct = ys.argmax()
    score_correct = preds[correct]

    loss = 0
    for i, pred in enumerate(preds):
        loss += max(0, pred + delta - score_correct)
    return loss

loss = []
reg_loss = []
print("Computing losses...")
for subset in subsets:
    preds = model.predict(subset[:, :11])
    my_preds = model2.predict(subset[:, :11])
    score = np.argmax(preds)
    best_wine = subset[score][11]
    # print("The best wine is:",best_wine)
    loss.append(svm_loss(preds, subset[:,11]))
    reg_loss.append(my_loss(my_preds, subset[:,11]))
    # print("Loss from Paper:", svm_loss(preds, subset[:,11]))
    # print("Regularized:",my_loss(my_preds, subset[:,11]))

plt.figure(figsize=(10,10))
plt.plot(loss, label="Given loss")
plt.plot(reg_loss, label="Regularized")
plt.legend()
plt.show()