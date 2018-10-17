Practical Machine Learning
========================================================
author: Peter Strauch
date: 17.10.2018
autosize: false
font-family: 'Helvetica'
transition: linear
transition-speed: fast

Data Science
========================================================
[Data Science Specialization](https://www.coursera.org/specializations/jhu-data-science),
John Hopkins University, Baltimore, USA (MD) --- Coursera (Beginner Specialization)

1. The Data Scientist's Toolbox (git, [GitHub](https://github.com/))
2. R Programming (syntax)
3. Getting and Cleaning Data (reading sources, ...)
4. Exploratory Data Analysis (graphs, ...)
5. Reproducible Research (dynamic document, ...)
6. Statistical Inference 
7. Regression Models
8. **Practical Machine Learning**
9. Developing Data Products (Rpresentation, ...)


Machine learning  (ML)
========================================================
Machine learning is a **data analytics technique** that teaches computers to do what comes naturally to humans and animals:  
+ learn from experience. 

Machine learning algorithms use **computational methods** to "learn" information directly **from data without relying on a predetermined equation** as a model.  
The algorithms adaptively improve their performance as the number of samples available for learning increases.

[Ref.](https://www.mathworks.com/discovery/machine-learning.html)


When Should You Use ML?
========================================================
Consider using ML when you have a **complex task**  
or a **large amount of data** and **lots of variables**,  
but **no existing formula or equation**.  

For example, ML is a good option if you need to handle situations like these:

+ Hand-written rules and equations are **too complex**  
(face recognition, speech recognition).
+ The rules of a **task are constantly changing**  
(fraud detection from transaction records).
+ The nature of the **data keeps changing**, and the program needs to adapt (automated trading, energy demand forecasting, predicting shopping trends).


How Machine Learning Works
========================================================
+ **supervised learning** -- trains model on known input, output  
(data: output ~ input, new data => predict future outputs)
+ **unsupervised learning** -- finds hidden patterns or intrinsic structures in input data.

![alt text](figures/ML_types.jpg)


Flow
========================================================
left: 27%

**Importance:**  
1. question  
2. input data  
3. features  
4. algorithm  
5. parameters  
6. evaluation
***
![alt text](figures/ML_process.jpg)


Importance: input data
========================================================
+ to have **correct data**
  1. **easy:** Best of all, we have data exactly what we want to predict (old video ratings to determine new video ratings)
  2. **harder**: We have data that suggests something that we want to predict (prediction of disease based on genes)
+ the **more similar data** are, the more accurate prediction is
+ **rather more data** than a better model


Importance: algorithm 
========================================================

1. **interpretable** -- for anyone
2. **simple** -- explainable
3. **precise** -- how interpretability and simplicity of the algorithm influences its accuracy
4. **quick** -- easy to make a model, and test it in small samples
5. **scalable** -- easy to apply to large data

you need to find a compromise: **1,2,4,5** vs. **3**


Common algorithms
========================================================
**Regression techniques**  -- predict continuous responses:  
+ linear model, 
+ nonlinear model, 
+ regularization, 
+ stepwise regression, 
+ decision trees - boosted and bagged trees, random forest,
+ neural networks, 
+ adaptive neuro-fuzzy learning, ...


Common algorithms
========================================================
**Classification techniques**  -- predict discrete responses:  
+ decision trees, 
  + boosted trees, 
  + bagged trees, 
      + random forest,
+ support vector machine (SVM), 
+ k-nearest neighbor, 
+ Naive Bayes, 
+ discriminant analysis, 
+ logistic regression, 
+ neural networks, ...


Decision trees
========================================================
left: 65%

+ to divide the outcome variable  
into 2 most homogenous groups
  + using 1 input variable (predictor)
+ the next splitting using another  
1 variable, ...

**Improvements:**  
+ boosted trees -- linear combination of predictors
+ bagged trees -- calculate on resamples, take the average
+ random forest

***
![alt text](figures/DecisionTree_ObamaClinton.jpg)
[https://goo.gl/c8rBHG](https://goo.gl/c8rBHG)


Problems with building a model
========================================================
left: 78%
![alt text](figures/overfitting.png)

***
especially trees

**Solution:**  
+ split data into test/train datasets



Prediction study design
========================================================

1. Define your **error rate** - generic error vs. other errors
2. **Split data** into:
  * training, testing, validation (optional)
3. **On the training set** pick features
4. **On the training set** pick prediction function $f(X)=Y$
6. If no validation 
  * Apply 1x to test set (only best model)
7. If validation
  * Apply to test set and refine
  * Apply 1x to validation set
  
  
  
Rules of thumb for predictions
========================================================
+ If you have a **large sample** size
    + 60% training
    + 20% testing
    + 20% validation
* If you have a **medium sample** size
    * 60% training
    * 40% testing
* If you have a **small sample** size
    * Do cross-validation
    * Report caveat of small sample size


Cross-validation
========================================================
left: 70%

**k-fold** (3-fold):
![alt text](figures/CV_k_fold.png)
***
**key idea:**

+ we don't have test set

**approach:**  
+ ...


Cross-validation
========================================================
left: 70%

**random sampling**:
![alt text](figures/CV_random_sampling.png)
***
**key idea:**

+ we don't have test set

**approach:**  

1. without replacement
2. with replacement = bootstrap



Common error measures
========================================================

**continuous outcome:**  
+ MSE, RMSE -- sensitive to outliers
+ median absolute deviation -- often more robust

**discrete outcome:**  
+ sensitivity (recall)
+ specificity
+ accuracy
+ concordance -- kappa, ...


Discrete metrices
========================================================
![alt text](figures/metrices.png)


Thank you for your attendance!
========================================================

Discussion

[pstrauch89@gmail.com]()  
[www.linkedin.com/in/peterstrauch/](https://www.linkedin.com/in/peterstrauch/)


