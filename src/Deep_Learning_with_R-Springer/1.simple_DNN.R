# Steps for simple NN :
# 1. Code the simple sigmoid activation function
# 2. Initialize the structural parameters w and b
# 3. Code the forward propagation function the labels and the cost
# 4. Predict the labels using structural parameters w and b
# 5. Reuse the parameters to backpropagate, to calculate the gradient of the structural parameters w and b

# Sigmoid function
sigmoid <- function(x){
  1/(1+exp(-x))
}

# Initializing weight and bias vector to zeros
initialize_with_zeros <- function(dim){
  w = matrix(0, nrow = dim, ncol = 1)
  b = 0
  return(list(w, b))
}

# Calculate cost and gradient
propagate <- function(w, b, X, Y){
  m = ncol(X)
  
  # Forward Propagation
  A = sigmoid((t(w) %*% X) + b)
  cost = (-1 / m) * sum(Y * log(A) + (1 - Y) * log(1 - A))
  
  # Backward Propagation
  dw = (1 / m) * (X %*% t(A - Y))
  db = (1 / m) * rowSums(A - Y)
  grads <- list(dw, db)
  
  return(list(grads, cost))
}

# Optimization using gradient descent algorithm
optimize <- function(w, b, X, Y, num_iter, learning_rate, print_cost = FALSE) {
  cost <- list()
  
  for (i in 1:num_iter) {
    # Cost and gradient calculation
    grads = propagate(w, b, X, Y)[[1]] # grads is a list
    cost[i] = propagate(w, b, X, Y)[[2]]
    
    # Retrieve the derivatives
    dw = matrix(grads[[1]])
    db = grads[[2]]
    
    # Update the parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # Record the cost
    if (i%%100 == 0) {
      costs <- cost
    }
    
    # Print the cost every 500th iteration
    if ((print_cost == T) & (i%%500 == 0)) {
      cat(sprintf("Cost after iteration %d: %06f\n", i,
                  cost[[i]]))
    }
    
    params <- list(w, b)
    grads <- list(dw, db)
  }
  
  return(list(params, grads, costs))
}

# Predict the output base on probability treshold
pred <- function(w, b, X) {
  m = ncol(X)
  Y_prediction <- matrix(0, nrow = 1, ncol = m)
  
  # Activation vector A to predict the probability of a dog/cat
  A = sigmoid((t(w) %*% X) + b)
  
  for (i in 1:ncol(A)) {
    if (A[1, i] > 0.5) {
      Y_prediction[1, i] = 1
    } else Y_prediction[1, i] = 0
  }
  
  return(Y_prediction)
}


# Simple Neural Network

simple_model <- function(X_train,
                         Y_train,
                         X_test,
                         Y_test,
                         num_iter,
                         learning_rate,
                         print_cost = FALSE){
  
# initialize parameters with zeros
w = initialize_with_zeros(nrow(X_train))[[1]]
b = initialize_with_zeros(nrow(X_train))[[2]]

# Gradient descent
optFn_output <- optimize(w,
                         b,
                         X_train,
                         Y_train,
                         num_iter,
                         learning_rate,
                         print_cost)

parameters <- optFn_output[[1]]
grads <- optFn_output[[2]]
costs <- optFn_output[[3]]

# Retrieve parameters w and b
w = as.matrix(parameters[[1]])
b = parameters[[2]]

# Predict test/train set examples
pred_train = pred(w, b, X_train)
pred_test = pred(w, b, X_test)

# Print train/test Errors
cat(sprintf("train accuracy: %#.2f \n", mean(pred_train == Y_train) * 100))
cat(sprintf("test accuracy: %#.2f \n", mean(pred_test == Y_test) * 100))

res = list("costs"= costs,
          "pred_train" = pred_train,
          "pred_test"= pred_test,
          "w" = w,
          "b" = b,
          "learning_rate" = learning_rate,
          "num_iter" = num_iter)

return(res)
}

