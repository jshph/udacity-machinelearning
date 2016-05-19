### concepts: Supervised learning:

- Discern among known labels and categories
- Assert categorization, machine learns the features that identify these categories

Data -> Decision surface

### Sklearn exercise in GaussianNB

```python
import numpy as np
X = np.array([[-1,-1], [-2, -1], [-3.-2], [1,1], [2,1], [3,2]])
Y = np.array([1,1,1,2,2,2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X,Y) # features, labels (same array size)
prediction = clf.predict([[-0.8,-1]]) # ask to fit this particular feature returns 1


# accuracy:
from sklearn.metrics import accuracy_score
ground_labels # let this be the correct classification.
# accuracy score is % correctly identified labels / total labels identified
print(accuracy_score(ground_labels, prediction))
```

### Bayes rule

- sensitivity: probability of true if condition C is true
- specificity: probability of true if condition C is false

Bayes rule:
prior probability + test evidence -> posterior probability

```
prior: P(C) = 0.01 = 1%
posterior:
P(C, positive) = P(C) * P(Pos|C)
P(~C, positive) = P(~C) * P(Pos | ~C) # total probability
```

Total probability, considering condition C and ~C, is a probability in the total space.
The above posterior is actually known as "joint probability". True posterior is obtained by dividing piece of the joint probabilities by the normalizer, which is P(C, Pos) + P(~C, Pos), total joint probability.

Why is Naive Bayes naive?
- Background: it uses hidden labels. We can only see what they do, but we have evidence of where they come from. Multiplying all these probabilities for activity from candidates A and B, we obtain a ratio of occurrences that help tell whether it's from A or B.
- Naive because the method counts frequency, but not order, of occurrences.

Naive Bayes advantages:
- very elegant, works well with large features spaces, easy to implement, efficient
disadvantages:
- breaks easily. Phrases encompassing multiple words get identified for their parts.

Guidelines:
- understand the theory of supervised learning without treating them like black boxes,
- use test data to present results
