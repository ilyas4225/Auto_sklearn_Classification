# Auto_sklearn_Classification
AutoSklearn is a powerful and efficient automated machine learning (AutoML) library. It simplifies the process of building machine learning models by automating tasks such as feature selection, hyperparameter tuning, and model selection.

The library is designed to handle various machine learning tasks, including classification, regression, and time series analysis.

In this project, we explore the power of Automated Machine Learning (AutoML) using the Breast Cancer and CIFAR-10 datasets. 

## Notebook description
In this repository the notebook ['[Breast_cancer.ipynb'](https://github.com/ilyas4225/Auto_sklearn_Classification/blob/main/Breast_cancer.ipynb)] contains the auto_sklearn image classifier for Breast Cancer dataset.


The Cifar10 notebook ['[Cifar10.ipynb'](https://github.com/ilyas4225/Auto_sklearn_Classification/blob/main/Cifar10.ipynb)] contains the auto sklearn classification with Cifar10 dataset.
## Findings
| Dataset      |    Classifier     |time-left-task | run_time/task|Accuracy_score|
|--------------|-------------------|---------------|--------------|--------------|
| Breast_cancer| sklearn classifier| 120           | 30           | 94.40%       |
| Cifar10     | sklearn classifier | 1000          | 100           | 46.45%       |



#### View the models found by auto-sklearn
Here is the obtained Models of auto-sklearn classification on breast _cancer dataset

```
          rank  ensemble_weight                type      cost  duration
model_id                                                               
7            1             0.06         extra_trees  0.014184  1.034909
27           2             0.06         extra_trees  0.014184  1.496269
16           6             0.06   gradient_boosting  0.021277  0.702333
21           3             0.04         extra_trees  0.021277  0.962193
30           5             0.02         extra_trees  0.021277  9.696474
35           4             0.04                 lda  0.021277  0.543221
2           10             0.04       random_forest  0.028369  1.152029
3           13             0.04                 mlp  0.028369  0.674715
10          12             0.02       random_forest  0.028369  1.289568
11          11             0.02       random_forest  0.028369  1.424863
19          14             0.08         extra_trees  0.028369  2.047359
22           8             0.04   gradient_boosting  0.028369  0.752507
44           9             0.06         extra_trees  0.028369  1.486794
48           7             0.02         extra_trees  0.028369  1.053900
5           16             0.02       random_forest  0.035461  1.345742
8           15             0.02       random_forest  0.035461  1.328252
31          17             0.04       random_forest  0.035461  1.093058
47          18             0.02                 lda  0.035461  0.543615
43          19             0.04  passive_aggressive  0.063830  0.581958
28          20             0.02        bernoulli_nb  0.070922  0.575956
32          21             0.04   gradient_boosting  0.078014  0.632487
39          22             0.02         extra_trees  0.099291  1.370216
38          23             0.02                 lda  0.127660  0.550949
36          24             0.14                 sgd  0.163121  0.547148
50          25             0.02                 qda  0.368794  0.894536

```

#### View the models found by auto-sklearn
```
{   2: {   'balancing': Balancing(random_state=1),
           'classifier': <autosklearn.pipeline.components.classification.ClassifierChoice object at 0x7fd148e92f10>,
           'cost': 0.028368794326241176,
           'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7fd148c75a30>,
           'ensemble_weight': 0.04,
           'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7fd148e92550>,
           'model_id': 2,
           'rank': 1,
           'sklearn_classifier': RandomForestClassifier(max_features=5, n_estimators=512, n_jobs=1,
                       random_state=1, warm_start=True)},
    3: {   'balancing': Balancing(random_state=1),
           'classifier': <autosklearn.pipeline.components.classification.ClassifierChoice object at 0x7fd148f32bb0>,
           'cost': 0.028368794326241176,
           'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7fd148ff1730>,
           'ensemble_weight': 0.04,
           'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7fd148f32760>,
           'model_id': 3,
           'rank': 2,
           'sklearn_classifier': MLPClassifier(activation='tanh', alpha=0.0001363185819149026, beta_1=0.999,
              beta_2=0.9, early_stopping=True,
              hidden_layer_sizes=(115, 115, 115),
              learning_rate_init=0.00018009776276177523, max_iter=32,
              n_iter_no_change=32, random_state=1, verbose=0, warm_start=True)},
    5: {   'balancing': Balancing(random_state=1, strategy='weighting'),
           'classifier': <autosklearn.pipeline.components.classification.ClassifierChoice object at 0x7fd148864a60>,
           'cost': 0.03546099290780147,
           'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7fd148bc0520>,
           'ensemble_weight': 0.02,
           'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7fd148b8cdf0>,
           'model_id': 5,
           'rank': 3,
           'sklearn_classifier': RandomForestClassifier(criterion='entropy', max_features=3, min_samples_leaf=2,
                       n_estimators=512, n_jobs=1, random_state=1,
    
 
```

