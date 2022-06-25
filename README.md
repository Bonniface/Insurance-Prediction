# Insurance-Prediction
![Img](https://zindi-public-release.s3.eu-west-2.amazonaws.com/uploads/competition/image/36/thumb_66aed506-93a2-43fd-b43b-924aa62784dc.jpeg)

# Description
This challenge was designed by Data Science Nigeria specifically for the DSN Bootcamp 2018, which takes place 19-24 November 2019. Welcome to the DSN participants!

After the Bootcamp, this competition will remain open to allow others in the Zindi community to learn and test their skills.

Description of the challenge:

Recently, there has been an increase in the number of building collapse in Lagos and major cities in Nigeria. Olusola Insurance Company offers a building insurance policy that protects buildings against damages that could be caused by a fire or vandalism, by a flood or storm.

You have been appointed as the Lead Data Analyst to build a predictive model to determine if a building will have an insurance claim during a certain period or not. You will have to predict the probability of having at least one claim over the insured period of the building.

The model will be based on the building characteristics. The target variable, Claim, is a:

1 if the building has at least a claim over the insured period.
0 if the building doesn’t have a claim over the insured period.


Python Package Link: https://pypi.org/project/BestClassificationModel/

Best Classification Model is used for supervised learning techniques where the target data is in binary form. It selects the best model from the seven classification model based on the accuracy. 

The seven classification model used in the given assignment are:

1. Logistic Regression
2. Naive Bayes
3. Stochastic Gradient Classifier
4. K Neighbors Classifier
5. Decision Tree Classifier
6. Random Forest Classifier
7. Support Vector Machine

#### User installation

If you already have a working installation of numpy, scipy and sklearn, the easiest way to install best-classification-model is using pip

#### `pip install BestClassificationModel`



#### Examples
```import

from Best_Classification_Model import best_model

import pandas

data = pandas.read_csv('Data.csv')

X = data.iloc[:, :-1]

Y = data['Class']

best_model, best_model_name, acc = best_model.bestClassificationModel(X, Y)

print(best_model)

print(best_model_name, ":", acc)```

`__Output__:
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

Random Forest:0.861145`

 
