# ML

PCA: pricial component analysis, use to reduce the dimension 

for linear regression: 

residual sum of squares: ((y_true - y_pred) ** 2).sum()
v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum()
Score: as (1 - u/v)


# steps:
##, StandardScaler(), to scaler your data
## drop the NA or duplicated rows
## cross-validation, split your train data into several CV/k-fold, and use each k-1 train data to train your model and do a combined average model
### Ridge regression:
#### min ||Xw-y||2(2) + alpha ||w||2
#### the greater the amount of shrinkage and thus the coefficients become more robust to collinearity
#### The ridge estimator is especially good at improving the least-squares estimate when multicollinearity is present.
### Lasso: estimatees sparse cofficients, sometime useful in some contexts due to its tendency to prefer solutions with fewer non-zero coefficients
#### min 1/2*n ||Xw-y||2 + alpha ||w||1
#### as alpha increases, bias increases, as alpha decreases, variance increases, this is L1 regularization, which adds a penalty equal to the absolute value of the magnitude of coefficients

### Elastic-Net
#### linear regression model trained with both l1 and l2-norm regulatzation of the coefficients
#### min 1/2*n ||Xw-y||2 + alpha ||w||1+alpha(1-p)/2* ||w||2

## try alpha in several numbers, use list to fit into the LassoCV command, for example:

```python
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)

```

#### generate polynomial combinations of the features with degree less than or euqal to the specified degree

#### [a,b], degree -2 polynomical features are [1,a,b,a^2, ab, b^2]

### GradientBoostingRegressor
#### learning_rate: shrinks the contribution of each tree by learning_reate.this is trade-off between learning_rate and n_estimators
#### n_estimatores: (default==100), the number of boosting stages to perform. Gradient boosting is fairly robust to over=fitting so a large number usually results in better performance
#### max_depth: max depth of the individual regression estimateros. The maximum depth limits the number of nodes in the tree.

```python
params = {'n_estimators': 800, 'max_depth': 1, 'min_samples_split': 2, 'learning_rate': .08, 'loss': 'ls', \
          'min_samples_leaf': 1}
classifier = ensemble.GradientBoostingRegressor(**params)
```

#### we can also use VotingClassifier to pick best one among relative simliar ML method

```
>>> from sklearn import datasets
>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.naive_bayes import GaussianNB
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.ensemble import VotingClassifier

>>> iris = datasets.load_iris()
>>> X, y = iris.data[:, 1:3], iris.target

>>> clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',
...                           random_state=1)
>>> clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
>>> clf3 = GaussianNB()

>>> eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

>>> for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
...     scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
...     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
Accuracy: 0.95 (+/- 0.04) [Logistic Regression]
Accuracy: 0.94 (+/- 0.04) [Random Forest]
Accuracy: 0.91 (+/- 0.04) [naive Bayes]
Accuracy: 0.95 (+/- 0.04) [Ensemble]
```

## recommended system
```
need to use recall and precision to test if the recommendation is good or not
based on content similarity, all user get similar date
based on user similarity, personzie the date
```
