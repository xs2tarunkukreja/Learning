# Binary Classification
Forward Pass or Forward Propagation
Backward Pass or Backward Propagation

## Problem Statement
Input = Image
Output = Is Cat? 1 or 0.

In image, we have 3 matrix - RGB
It converted into feature vector - 1 D - first all row one by one for R then G and then B.

X = Nx * m 
m is number of observations. N is feature vector length = 64 * 64 * 3.
Y shape = (1, M) 

# Logistic Regression
Used when o/p is either 0 or 1.

![alt text](image-3.png)

# LR Cost Function
![alt text](image-4.png)

# Gradient Descent
We decide w and b initial value. 
Gradient Descent - We try to update w and b so that we can reach to lower of cost function.
![alt text](image-5.png)

## Understand
Ignore b for now.
Derivation is slope of a function.

![alt text](image-6.png)

# Derivatives
![alt text](image-7.png)

If we increase x very little then how much y increase.. f(x) = 3x
slope = derivative = 3. 

It is always 3 independent of x.

## Complex Example instead of linear.
f(x) = x^2
Derivative keep changing based on value of x.
![alt text](image-8.png)


# Computation Graph
Computation of Neural Network
1. Forward Pass/Propogation to calculate the output.
2. Followed by Backward pass that use gradient descent or derivative.
![alt text](image-9.png)

# Derivatives with Computation Graph
![alt text](image-10.png)
![alt text](image-11.png)
![alt text](image-12.png)

# Logistic Regression Gradient Descent
Logistic Regression
![alt text](image-13.png)
![alt text](image-14.png)

Find Derivative of Lost Function with respect to weigths and b
![alt text](image-15.png)

![alt text](image-16.png)

# Gradient Descent on m Examples