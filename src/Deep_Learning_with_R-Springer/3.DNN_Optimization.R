# import function
source("./src/Deep_Learning_with_R-Springer/2.Deep_Neural_Networks_1.R")

# Mini batch GD function
random_mini_batches <- function(X, Y, mini_batch_size, seed){
  set.seed(seed)
  # Get number of training samples
  m = dim(X)[2]
  # Initialize mini batches
  mini_batches = list()
  # Create a list of random numbers
  rand_sample = c(sample(m))
  # Randomly shuffle the training data
  shuffled_X = X[, rand_sample]
  shuffled_Y = Y[rand_sample]
  # Compute number of mini batches
  num_minibatches = floor(m / mini_batch_size)
  batch = 0
  for(i in 0:(num_minibatches - 1)){
    batch = batch + 1
    # Set the lower & upper bound of the mini batches
    lower = (i * mini_batch_size) + 1
    upper = ((i + 1) * mini_batch_size)
    mini_batch_X = shuffled_X[, lower:upper]
    mini_batch_Y = shuffled_Y[lower:upper]
    mini_batch = list("mini_batch_X" = mini_batch_X,
                      "mini_batch_Y" = mini_batch_Y)
    mini_batches[[batch]] = mini_batch
  }
  
  # If the batch size does not divide evenly with mini batch size
  if(m %% mini_batch_size != 0){
    # Set the start and end of last batch
    start = floor(m / mini_batch_size) * mini_batch_size
    end = start + m %% mini_batch_size
    mini_batch_X = shuffled_X[, (start + 1):end]
    mini_batch_Y = shuffled_Y[(start + 1):end]
    mini_batch_last = list("mini_batch_X" = mini_batch_X,
                           "mini_batch_Y" = mini_batch_Y)
    mini_batches[[batch + 1]] <- c(mini_batch, mini_batch_last)
  }
  return(mini_batches)
}

# Momentum update function
initialize_velocity <- function(parameters) {
  L = length(parameters)
  v = list()
  for (layer in 1:L) {
    v[[paste("dW", layer, sep = "")]] = 0 * parameters[[paste("W",
        layer, sep = "")]]
    v[[paste("db", layer, sep = "")]] = 0 * parameters[[paste("b",
        layer, sep = "")]]
  }
  
  return(v)
}

update_params_with_momentum <- function(parameters, gradients,
  velocity, beta, learning_rate) {
  L = length(parameters)/2
  for (l in 1:L) {
    velocity[[paste("dW", l, sep = "")]] = beta * velocity[[paste("dW",
        l, sep = "")]] + (1 - beta) * gradients[[paste("dW",
        l, sep = "")]]
    velocity[[paste("db", l, sep = "")]] = beta * velocity[[paste("db",
        l, sep = "")]] + (1 - beta) * gradients[[paste("db",
        l, sep = "")]]
    parameters[[paste("W", l, sep = "")]] = parameters[[paste("W",
        l, sep = "")]] - learning_rate * velocity[[paste("dW",
        l, sep = "")]]
    parameters[[paste("b", l, sep = "")]] = parameters[[paste("b",
        l, sep = "")]] - learning_rate * velocity[[paste("db",
        l, sep = "")]]
  }
  
  return(list(parameters = parameters, Velocity = velocity))
}

# Adam function
initialize_adam <- function(parameters) {
  
  L = length(parameters)/2
  v = list()
  s = list()
  
  for (layer in 1:L) {
    v[[paste("dW", layer, sep = "")]] = 0 * parameters[[paste("W",
        layer, sep = "")]]
    v[[paste("db", layer, sep = "")]] = 0 * parameters[[paste("b",
        layer, sep = "")]]
    s[[paste("dW", layer, sep = "")]] = 0 * parameters[[paste("W",
        layer, sep = "")]]
    s[[paste("db", layer, sep = "")]] = 0 * parameters[[paste("b",
        layer, sep = "")]]
  }
  
  return(list(V = v, S = s))
}

update_params_with_adam <- function(parameters, gradients, v,
                                    s, t, beta1, beta2, learning_rate, epsilon) {
  L = length(parameters)/2
  v_corrected = list()
  s_corrected = list()
  
  for (layer in 1:L) {
    v[[paste("dW", layer, sep = "")]] = beta1 * v[[paste("dW",
        layer, sep = "")]] + (1 - beta1) * (gradients[[paste("dW",
        layer, sep = "")]])
    v[[paste("db", layer, sep = "")]] = beta1 * v[[paste("db",
        layer, sep = "")]] + (1 - beta1) * gradients[[paste("db",
        layer, sep = "")]]
    
    v_corrected[[paste("dW", layer, sep = "")]] = v[[paste("dW",
        layer, sep = "")]]/(1 - beta1^t)
    v_corrected[[paste("db", layer, sep = "")]] = v[[paste("db",
        layer, sep = "")]]/(1 - beta1^t)
    
    s[[paste("dW", layer, sep = "")]] = beta2 * s[[paste("dW",
        layer, sep = "")]] + (1 - beta2) * (gradients[[paste("dW",
        layer, sep = "")]])^2
    s[[paste("db", layer, sep = "")]] = beta2 * s[[paste("db",
        layer, sep = "")]] + (1 - beta2) * (gradients[[paste("db",
        layer, sep = "")]])^2
    
    s_corrected[[paste("dW", layer, sep = "")]] = s[[paste("dW",
        layer, sep = "")]]/(1 - beta2^t)
    s_corrected[[paste("db", layer, sep = "")]] = s[[paste("db",
        layer, sep = "")]]/(1 - beta2^t)
    
    parameters[[paste("W", layer, sep = "")]] = parameters[[paste("W",
        layer, sep = "")]] - learning_rate * (v_corrected[[paste("dW",
        layer, sep = "")]])/(sqrt(s_corrected[[paste("dW",
        layer, sep = "")]]) + epsilon)
    parameters[[paste("b", layer, sep = "")]] = parameters[[paste("b",
        layer, sep = "")]] - learning_rate * (v_corrected[[paste("db",
        layer, sep = "")]])/(sqrt(s_corrected[[paste("db",
        layer, sep = "")]]) + epsilon)
  }
  
  return(list(parameters = parameters, Velocity = v, S = s))
}

# Model
DNN <- function(X,
                  Y,
                  X_test,
                  Y_test,
                  layers_dims,
                  hidden_layer_act,
                  output_layer_act,
                  optimizer,
                  learning_rate,
                  mini_batch_size,
                  num_epochs,
                  initialization,
                  beta,
                  beta1,
                  beta2,
                  epsilon,
                  print_cost = F){
  costs <- NULL
  t = 0
  set.seed = 1
  seed = 10
  parameters = initialize_params(layers_dims, initialization)
  v = initialize_adam(parameters)[["V"]]
  s = initialize_adam(parameters)[["S"]]
  
  velocity = initialize_velocity(parameters)
  
  start_time <- Sys.time()
  
  for(i in 0:num_epochs){
    seed = seed + 1
    minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
    
    for(batch in 1:length(minibatches)){
      mini_batch_X = (minibatches[[batch]][['mini_batch_X']])
      mini_batch_Y = minibatches[[batch]][['mini_batch_Y']]
    
      AL = forward_prop(mini_batch_X, parameters, hidden_layer_act,
                        output_layer_act)[['AL']]
      
      caches = forward_prop(mini_batch_X, parameters, hidden_layer_act,
                            output_layer_act)[['caches']]
      
      cost <- compute_cost(AL, mini_batch_X, mini_batch_Y, num_classes = 0,
                           output_layer_act)
      
      gradients = back_prop(AL, mini_batch_Y, caches, hidden_layer_act,
                            output_layer_act)
      
      if(optimizer == 'gd'){
        parameters = update_params(parameters, gradients, learning_rate)
      }
      else if(optimizer == 'momentum'){
        parameters = update_params_with_momentum(parameters,
                                                 gradients,
                                                 velocity,
                                                 beta,
                                                 learning_rate
                                                 )[["parameters"]]
        velocity = update_params_with_momentum(parameters,
                                               gradients,
                                               velocity,
                                               beta,
                                               learning_rate
                                               )[["Velocity"]]
      }
      else if(optimizer == 'adam'){
        t = t + 1
        parameters = update_params_with_adam(parameters,
                                             gradients,
                                             v,
                                             s,
                                             t,
                                             beta1,
                                             beta2,
                                             learning_rate,
                                             epsilon
                                             )[["parameters"]]
        v = update_params_with_adam(parameters,
                                    gradients,
                                    v,
                                    s,
                                    t,
                                    beta1,
                                    beta2,
                                    learning_rate,
                                    epsilon
                                    )[["Velocity"]]
        s = update_params_with_adam(parameters,
                                    gradients,
                                    v,
                                    s,
                                    t,
                                    beta1,
                                    beta2,
                                    learning_rate,
                                    epsilon
                                    )[["S"]]
      }
    }
    
    if(print_cost == T & i %% 1000 == 0){
      print(paste0("Cost after iteration " , i, ' = ', cost, sep = ' '))
    }
    if(print_cost == T & i %% 100 == 0){
      costs = c(costs, cost)
    }
  }
  
  Y_prediction_train = predict_model(parameters, X, hidden_layer_act,
                                     output_layer_act)
  Y_prediction_test = predict_model(parameters, X_test, hidden_layer_act,
                                    output_layer_act)
  
  cat(sprintf("train accuracy: %05f, \n",
              (100 - mean(abs(Y_prediction_train - Y)) * 100)))
  cat(sprintf("test accuracy: %05f, \n",
              (100 - mean(abs(Y_prediction_test - Y_test)) * 100)))
  cat(sprintf("Cost after: %d, iterations is: %05f, \n", i, cost))
  end_time <- Sys.time()
  cat(sprintf("Application running time: %#.3f minutes",
              end_time - start_time ))
  
  return(list("parameters" = parameters, "costs" = costs))
}


