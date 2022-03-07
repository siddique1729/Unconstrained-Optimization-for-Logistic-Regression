# Unconstrained-Optimization-for-Logistic-Regression
Solving unconstrained optimization for determining bridge class using Nelder-Mead &amp; Newton's Method

# Question 1

Fitting a logistic regression model by selecting between 4 – 6 inputs from the variables provided. The output variable is the bridge condition (which is binary).

## Part 1 - Importing Required Libraries

- Input Work

<img src="images/img1.png">

## Part 2 - Checking directory, reading and viewing the dataset 

- Input Work
<img src="images/img2.png">

- Output
<img src="images/out1.png">
<img src="images/out2.png">

## Part 3 - Data Cleaning

- Input Work
<img src="images/img3.png">
<img src="images/img4.png">

- Output Work
<img src="images/out3.png">
<img src="images/out4.png">

## Part 4 - Normalization and Feature Selection

- Input Work
<img src="images/img5.png">
<img src="images/img6.png">

- Output
<img src="images/out5.png">
<img src="images/out6.png">

## Part 5 - Writing the logistic function

- Input Work
<img src="images/img7.png">

- Mathematical equation of written logistic function
<img src="images/logreg.png">

## Part 6 - Optimization using Nelder-Mead

- Input Work
<img src="images/img8.png">

- Output
<img src="images/Neldermead.png">

## Part 7 - Optimization using Newton's Method

- Input Work
<img src="images/img9.png">

- Output
<img src="images/newton.png">

## Part 8 - Checking optimality of the function

- Input Work
<img src="images/img10.png">

- It is a convex function with a global minima
- The eigen values are: 3.22444850e+02, 4.45930376e+01, 1.58613226e+01, 7.67344450e+00, 2.06681472e+00, 2.36034727e-01, 8.55399546e-01

# Question 2

To write a program to develop the 95% Confidence Levels for each of the estimated Model Coefficient of the Logistic Regression Model using the Bootstrap Approach

The code is similar from Part 1 - 4

## Random Sampling with Repetition

<img src="images/img21.png">
<img src="images/img22.png">

## Computing 95% confidence interval for the coefficients for 5000 samples

<img src="images/img23.png">
<img src="images/img24.png">

- Formula for calculating confidence interval
<img src="images/conf.png">
