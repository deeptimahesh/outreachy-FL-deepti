# outreachy-FL-deepti
### A detailed look into loss functions and optimization methods

I have had a thorough look into the github as well as the documentation provided for the project. With some insights as to the data collection done by Firefox as well as a revision of knowledge of SVMs, margin maximizing loss functions and other optimization methods, these are some improvements which can possibly be carried out.

The loss function used as of now remains a hinge loss for pointwise-ranking which is to be minimized:

        def svm_loss(preds, ys, delta=0):
            correct = ys.argmax()
            score_correct = preds[correct]
            
            loss = 0
            
            for i, pred in enumerate(preds):
                loss += max(0, pred + delta - score_correct)            
                    
            return loss

## Case I: L2 Regularization or Ridge Regularization
Is it possible that the penalties for scores which are higher than the user-selected one (ie on the wrong side of the separating hyperplane) must be penalized further? Regularization may be the solution for this. Regularization adds a term to the loss function of the problem.

L2 regularization forces the weights to be small but does not make them zero and does non sparse solution. However, it is not nobust to outliers. In this case, outliers donot make much of a difference, since if a user tries to visit a webpage he rarely visits / has not visited, he's most likely to type out the address himself.

Ridge regression performs better when all the input features influence the output and all with weights are of roughly equal size, which is a case which might happen often here.

If zero-weights are desired, we can use elastic net which is a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods. Moreover Elastic Net can be reduced to the linear support vector machine.

^ More on this to be added

### **__Doubt:__** Number of considered visits (10): If this number increases too much, it would hurt performance. (Nevermind, figured this out) <- Look at this variable later properly

## Case II

The variables (22) which are to be optimized are: 
  - firstBucketCutoff
  - secondBucketCutoff
  - thirdBucketCutoff
  - fourthBucketCutoff
  - firstBucketWeight
  - secondBucketWeight
  - thirdBucketWeight
  - fourthBucketWeight
  - defaultBucketWeight
  - embedVisitBonus
  - framedLinkVisitBonus
  - linkVisitBonus
  - typedVisitBonus
  - bookmarkVisitBonus
  - downloadVisitBonus
  - permRedirectVisitBonus
  - tempRedirectVisitBonus
  - redirectSourceVisitBonus
  - defaultVisitBonus
  - unvisitedBookmarkBonus
  - unvisitedTypedBonus
  - reloadVisitBonu

The optimization process is started from the current set of values and then improved from there on.

Some other variables which can be utilized without modifying any prior data collection methods are:

- s
- d

## Case III: Frecency Modification (maybe out of scope)
Is it possible to perhaps get time spent on the webpage? For example, Facebook is a website on which people spend a lot of time on. Might be possible to add that to the frecency score.

## Case IV: Delta values