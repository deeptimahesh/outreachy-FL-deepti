# outreachy-FL-deepti

## A detailed look into loss functions and optimization methods

I have had a thorough look into the github as well as the documentation provided for the project. With some insights as to the data collection done by Firefox as well as a revision of knowledge of SVMs, margin maximizing loss functions and other optimization methods, these are some improvements which can possibly be carried out.

The loss function used as of now remains a hinge loss for pointwise-ranking which is to be minimized:

    def svm_loss(preds, ys, delta=0):
        correct = ys.argmax()
        score_correct = preds[correct]
        
        loss = 0
        
        for i, pred in enumerate(preds):
            loss += max(0, pred + delta - score_correct)            
                
        return loss

### __Case I: L2 Regularization or Ridge Regularization__

Is it possible that the penalties for scores which are higher than the user-selected one (ie on the wrong side of the separating hyperplane) must be penalized further? Regularization may be the solution for this. Regularization adds a term to the loss function of the problem.

L2 regularization forces the weights to be small but does not make them zero and thus, a non sparse solution is obtained. However, it is not robust to outliers. In this case, outliers donot make much of a difference, since if a user tries to visit a webpage he rarely visits / has not visited, he's most likely to type out the address himself.

Ridge regression performs better when all the input features influence the output and all with weights are of roughly equal size, which is a case which might happen often here.

If zero-weights are desired, we can use elastic net which is a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods. Moreover Elastic Net can be reduced to the linear support vector machine.

#### Edits Made

1. __Why Regularization?__
    * With larger and more complex models such as what this problem uses, overfitting is a known consequence. Regularization attempts to reduce the variance of the estimator by simplifying it, which increases the bias, in such a way that the expected error decreases. Moeover, the number of samples / user data is small.

      The regularisation terms are constraints by which an optimisation algorithm must adhere to when minimising the loss function, apart from doing what was being optimized before.
    * Penalize weights that are large such that more bias is not given to some bonuses which would lead to very incorrect answers as we do not want to assume that the user for example only visits bookmarked sites. At some point, the penalty of having too large will outweigh whatever gain made in the loss function.

2. __Ridge Regularization and User Behavior__
    * A lot of our bonuses might alerady be zero since most users do not typically bookmark some websites, etc. Thus, we regularize it in such a way that __no other bonuses is reduced to a zero__. L2 regularization is used.
    * Moreover we do not have a very large number of weights which also makes the loss function not too complex. Thus, Lasso / L1 may seem like a bit of overkill.
    * Moreover L1 is influenced only by sign of the weights whereas L2 is affected by magnitude, and doubling of the regularization parameter (during gradient descent) and since our weignts are never assigned a non-zero value (we never want to assume that the users will NOT visit a site),, L2 is considered a better solution in this case.

### __Case II__ (ruled as out of scope)

The variables (22) which are to be optimized are:

* firstBucketCutoff
* secondBucketCutoff
* thirdBucketCutoff
* fourthBucketCutoff
* firstBucketWeight
* secondBucketWeight
* thirdBucketWeight
* fourthBucketWeight
* defaultBucketWeight
* embedVisitBonus
* framedLinkVisitBonus
* linkVisitBonus
* typedVisitBonus
* bookmarkVisitBonus
* downloadVisitBonus
* permRedirectVisitBonus
* tempRedirectVisitBonus
* redirectSourceVisitBonus
* defaultVisitBonus
* unvisitedBookmarkBonus
* unvisitedTypedBonus
* reloadVisitBonu

The optimization process is started from the current set of values and then improved from there on.

Some other variables which can be utilized without modifying any prior data collection methods are:

* Bonus for Already Open Tabs
* Bonus for session lengths

[This link](https://github.com/mozilla/legal-docs/blob/master/firefox_privacy_notice/en-US.md) confirms that session lengths are also collected which might prove to be useful with respect to suggesting better predictions when utilized with frecency.

`< Need to look into more >`

Gather time-related statistics based on training with more number of considered visits (ie, greater than 10 or if lesser number is also enough)?

### __Case III: Delta values__ (out of scope?)

The loss function as of now tries to maximize the margin between correct classification and all other wrong classifications. This means the decision boundary (delta here) tries to be as furthest away from the nearest user-selected output.

This is nothing but the c parameter in SVMs. C is a regularization parameter that controls the trade off between the achieving a low training error and a low testing error that is the ability to generalize your classifier to unseen data.

Optimizing c involves cross-validation, etc. Would this be a useful thing to add/pursue?

## Note to the mentors

Hey!
I'm Deepti Mahesh, an Outreachy applicant from India.
Looking forward to learning and contributing as much as I can.
Thanks!

## References

* Provided Links: <https://florian.github.io/federated-learning-firefox/>
* Github: <https://github.com/florian/federated-learning/>
