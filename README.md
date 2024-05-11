# Gradient Descent (GD) Optimizer

## Introduction

The provided class implements a gradient descent (GD) optimizer, which is commonly used for training machine learning models. It supports mini-batch gradient descent, stochastic gradient descent (SGD), momentum-based GD, and Nesterov Accelerated Gradient (NAG).

## Usage

To use the optimizer, instantiate an object of the class `GDOptimizer` with appropriate parameters. The key parameters include:
- `learning_rate`: The learning rate determines the step size taken in the direction opposite to the gradient.
- `batch_size`: The batch size determines the number of samples used for each update step. Setting it to 1 enables stochastic gradient descent (SGD), while setting it to the length of the data enables full-batch gradient descent.
- `accelerated`: This parameter enables acceleration options. Setting it to 0 disables acceleration. Setting it to 1 activates momentum-based GD, and setting it to 2 activates Nesterov Accelerated Gradient (NAG).

The provided `BFGS` function implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization algorithm for minimizing a given objective function. It takes the following parameters:
- `gradient`: The gradient function to compute the gradient of the objective function.
- `LR`: The learning rate or step size for updating the model parameters.
- `theta_curr`: The initial guess for the optimal model parameters.
- `theta_prev`: The previous model parameters.
- `X`: The input data matrix.
- `y`: The target values.
- `num_iter`: The maximum number of iterations (default is 100).
- `epsilon`: The convergence threshold (default is 0.001).
