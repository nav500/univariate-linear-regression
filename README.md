# Univariate linear regression

## Introduction

Implementaion of univariate linear regression in python.

The program uses gradient descent algorithm to train the model.

The data set contains single feature - size of house in 1000 sq feets. The target value is - price of the house in 1000s of dollars.

The output of model $ f $, estimated target value, is defined in the following equation :-

$$ \hat{y} = f(x) = wx+b $$

The cost function $ J $ is given by the following equation :- 

$$ J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$ 

## How to execute

use the following command in the root folder (univariate-linear-regression) of the program :-

```python
python src/main.py
```