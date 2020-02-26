# import function
source("./src/Deep_Learning_with_R-Springer/3.DNN_Optimization.R")

# Foward propagation function with regularization
forward_prop_Reg <- function(X,
                             parameters,
                             hidden_layer_act,
                             output_layer_act,
                             keep_prob){
  dropout_matrix <- list()
  caches <- list()
  A <- X
  L <- length(parameters) / 2
  
  for(l in 1:(L - 1)){
    A_prev <- A
    W <- parameters[[paste("W", l, sep = "")]]
    b <- parameters[[paste("b", l, sep = "")]]
    actForward <- f_prop_helper(A_prev, W, b, hidden_layer_act[[l]])
    A <- actForward[['A']]
    caches[[l]] <-actForward
    
    # Randomly drop some activation units
    # Create a matrix with the same shape as A
    set.seed(1)
    i = dim(A)[1]
    j = dim(A)[2]
    k <- rnorm(i * j)
    # Convert values in k to between 0 and 1
    k = (k - min(k)) / (max(k) - min(k))
    # Create a matrix of D
    D <- matrix(k, nrow = i, ncol = j)
    # Find D which is less than equal to keep_prob
    D <- D < keep_prob
    # Shut down those neurons which are less than keep_prob
    A <- A * D
    # Scale the value of neurons that have not been
    # shut down to keep the expected values
    A <- A / keep_prob
    dropout_matrix[[paste("D", l, sep = "")]] <- D
  }
  
  W <- parameters[[paste("W", L, sep = "")]]
  b <- parameters[[paste("b", L, sep = "")]]
  
  actForward = f_prop_helper(A, W, b, output_layer_act)
  AL <- actForward[['A']]
  caches[[L]] <- actForward
  
  return(list("AL" = AL, "caches" = caches, "dropout_matrix" = dropout_matrix))
}

# Cost function with regularization
compute_cost_with_Reg <- function(AL, X, Y, num_classes, parameters,
    lambd, output_layer_act) {
  
  # Cross-Entropy cost
  if (output_layer_act == "sigmoid") {
    m = length(Y)
    cross_entropy_cost = -(1/m) * sum(Y * log(AL) + (1 -
        Y) * log(1 - AL))
  } else if (output_layer_act == "softmax") {
    m = ncol(X)
    y.mat <- matrix(Y, ncol = 1)
    y <- matrix(0, nrow = m, ncol = num_classes)
    for (i in 0:(num_classes - 1)) {
      y[y.mat[, 1] == i, i + 1] <- 1
    }
    correct_logprobs <- -log(AL)
    cross_entropy_cost <- sum(correct_logprobs * y)/m
  }
  
  # Regularization cost
  L <- length(parameters)/2
  L2_Reg_Cost = 0
  
  for (l in 1:L) {
    L2_Reg_Cost = L2_Reg_Cost + sum(parameters[[paste("W",
        l, sep = "")]]^2)
  }
  L2_Reg_Cost = lambd/(2 * m) * L2_Reg_Cost
  cost = cross_entropy_cost + L2_Reg_Cost
  
  return(cost)
}

# Backward propagation with regularization
back_prop_Reg_helper <- function(dA,
                                 cache,
                                 X,
                                 Y,
                                 num_classes,
                                 hidden_layer_act,
                                 lambd){
  
  forward_cache <-cache[['forward_cache']]
  activation_cache <- cache[['activation_cache']]
  A_prev <- forward_cache[['A_prev']]
  m = dim(A_prev)[2]
  activation_cache <- cache[['activation_cache']]
  
  if(hidden_layer_act == "relu"){
    dZ <- derivative_relu(dA, activation_cache)
  }
  else if(hidden_layer_act == "sigmoid"){
    dZ <- derivative_sigmoid(dA, activation_cache)
  }
  else if(hidden_layer_act == "tanh"){
    dZ <- derivative_tanh(dA, activation_cache)
  }
  else if(hidden_layer_act == "softmax"){
    dZ <- derivative_softmax(dAL, activation_cache, X, Y, num_classes)
  }
  
  W <- forward_cache[['W']]
  b <- forward_cache[['b']]
  m = dim(A_prev)[2]
  
  if(hidden_layer_act == 'softmax'){
    dW = 1 / m * t(dZ) %*% t(A_prev) + (lambd / m) * W
    db = 1 / m * colSums(dZ)
    dA_prev = t(W) %*% t(dZ)
  }
  else{
    dW = 1 / m * dZ %*% t(A_prev) + (lambd / m) * W
    db = 1 / m * rowSums(dZ)
    dA_prev = t(W) %*% dZ
  }
  
  return(list("dA_prev" = dA_prev, "dW" = dW, "db" = db))
}

back_prop_Reg <- function(AL,
                          X,
                          Y,
                          num_classes,
                          caches,
                          hidden_layer_act,
                          output_layer_act,
                          keep_prob,
                          dropout_matrix,
                          lambd){
  
  gradients = list()
  L = length(caches)
  m = dim(AL)[2]
  
  if(output_layer_act == "sigmoid"){
    dAL = -((Y/AL) - (1 - Y)/(1 - AL))
  }
  else if(output_layer_act == 'softmax') {
    y.mat <- matrix(Y, ncol = 1)
    y <- matrix(0, nrow=ncol((X)), ncol = num_classes)
    for (i in 0:(num_classes - 1)) {
      y[y.mat[, 1] == i,i+1] <- 1
    }
    dAL = (AL - y)
  }
  
  current_cache = caches[[L]]$cache
  
  if(lambd == 0){
    loop_back_reg_vals <- back_prop_Reg_helper(dAL,
                                               current_cache,
                                               X, Y,
                                               num_classes,
                                               hidden_layer_act =
                                                 output_layer_act,
                                               lambd)
  }
  else {
    loop_back_reg_vals = back_prop_Reg_helper(dAL, current_cache,
                                              X, Y,
                                              num_classes,
                                              hidden_layer_act =
                                              output_layer_act,
                                              lambd)
  }
  
  if(output_layer_act == "sigmoid"){
    gradients[[paste("dA", L, sep = "")]] <- loop_back_reg_vals[['dA_prev']]
  }
  else if(output_layer_act == "softmax"){
    gradients[[paste("dA", L, sep = "")]] <- (loop_back_reg_vals[['dA_prev']])
  }
  
  gradients[[paste("dW", L, sep = "")]] <- loop_back_reg_vals[['dW']]
  gradients[[paste("db", L, sep = "")]] <- loop_back_reg_vals[['db']]
  
  for(l in (L-1):1){
    current_cache = caches[[l]]$cache
    if (lambd == 0 & keep_prob < 1){
      D <- dropout_matrix[[paste("D", l, sep = "")]]
      # Multiply gradient with dropout matrix &
      # divide by keep_prob to keep expected value same
      gradients[[paste('dA', l + 1, sep = "")]] =
        gradients[[paste('dA', l + 1, sep = "")]] * D / keep_prob
      loop_back_vals <- back_prop_Reg_helper(gradients[[paste('dA',
                                                               l + 1,
                                                               sep = "")]],
                                             current_cache,
                                             X,
                                             Y,
                                             num_classes,
                                             hidden_layer_act[[l]],
                                             lambd)
    }
    else if(lambd != 0 & keep_prob == 1){
      loop_back_vals = back_prop_Reg_helper(gradients[[paste('dA',
                                                              l + 1,
                                                              sep = "")]],
                                            current_cache,
                                            X,
                                            Y,
                                            num_classes,
                                            hidden_layer_act[[l]],
                                            lambd)
    }
    else if(lambd == 0 & keep_prob == 1){
      loop_back_vals = back_prop_Reg_helper(gradients[[paste('dA',
                                                              l + 1,
                                                              sep = "")]],
                                            current_cache,
                                            X,
                                            Y,
                                            num_classes,
                                            hidden_layer_act[[l]],
                                            lambd = 0)
    }
    
    gradients[[paste("dA", l, sep = "")]] <- loop_back_vals[['dA_prev']]
    gradients[[paste("dW", l, sep = "")]] <- loop_back_vals[['dW']]
    gradients[[paste("db", l, sep = "")]] <- loop_back_vals[['db']]
  }
  
  return(gradients)
}

DNN_reg <- function(X,
                           Y,
                           X_test,
                           Y_test,
                           num_classes,
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
                           keep_prob,
                           lambd,
                           verbose = F){
  
  start_time <- Sys.time()
  costs <- NULL
  converged = FALSE
  param <- NULL
  t = 0
  iter = 0
  set.seed = 1
  seed = 10
  parameters = initialize_params(layers_dims, initialization)
  v = initialize_adam(parameters)[["V"]]
  s = initialize_adam(parameters)[["S"]]
  velocity = initialize_velocity(parameters)
  
  for(i in 0:num_epochs){
    seed = seed + 1
    iter = iter + 1
    minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
    
    for(batch in 1:length(minibatches)){
      mini_batch_X = (minibatches[[batch]][['mini_batch_X']])
      mini_batch_Y = minibatches[[batch]][['mini_batch_Y']]
    
        if(keep_prob == 1){
        AL = forward_prop(mini_batch_X,
                          parameters,
                          hidden_layer_act,
                          output_layer_act)[['AL']]
        caches = forward_prop(mini_batch_X,
                              parameters,
                              hidden_layer_act,
                              output_layer_act)[['caches']]
      }
      else if(keep_prob < 1){
        AL = forward_prop_Reg(mini_batch_X,
                              parameters,
                              hidden_layer_act,
                              output_layer_act,
                              keep_prob)[['AL']]
        caches = forward_prop_Reg(mini_batch_X,
                                  parameters,
                                  hidden_layer_act,
                                  output_layer_act,
                                  keep_prob)[['caches']]
        dropout_matrix = forward_prop_Reg(mini_batch_X,
                                          parameters,
                                          hidden_layer_act,
                                          output_layer_act,
                                          keep_prob)[['dropout_matrix']]
      }
      
      # Compute Cost
      cost <- compute_cost_with_Reg(AL,
                                    mini_batch_X,
                                    mini_batch_Y,
                                    num_classes,
                                    parameters,
                                    lambd,
                                    output_layer_act)
      
      # Backward propagation
      if(lambd == 0 & keep_prob == 1){
        gradients = back_prop_Reg(AL,
                                  mini_batch_X,
                                  mini_batch_Y,
                                  num_classes,
                                  caches,
                                  hidden_layer_act,
                                  output_layer_act,
                                  keep_prob = 1,
                                  dropout_matrix = NULL,
                                  lambd = 0)
      }
      else if(lambd != 0 & keep_prob == 1){
        gradients = back_prop_Reg(AL,
                                  mini_batch_X,
                                  mini_batch_Y,
                                  num_classes,
                                  caches,
                                  hidden_layer_act,
                                  output_layer_act,
                                  keep_prob = 1,
                                  dropout_matrix = NULL,
                                  lambd)
      }
      else if(lambd == 0 & keep_prob < 1){
        gradients = back_prop_Reg(AL,
                                  mini_batch_X,
                                  mini_batch_Y,
                                  num_classes,
                                  caches,
                                  hidden_layer_act,
                                  output_layer_act,
                                  keep_prob = 1,
                                  dropout_matrix,
                                  lambd = 0)
      }
      
      if(optimizer == 'gd'){
        parameters = update_params(parameters, gradients, learning_rate)
      }
      else if(optimizer == 'momentum'){
        parameters = update_params_with_momentum(parameters,
                                                 gradients,
                                                 velocity,
                                                 beta,
                                                 learning_rate)[["parameters"]]
        velocity = update_params_with_momentum(parameters,
                                               gradients,
                                               velocity,
                                               beta,
                                               learning_rate)[["Velocity"]]
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
                                             epsilon)[["parameters"]]
        v = update_params_with_adam(parameters,
                                    gradients,
                                    v,
                                    s,
                                    t,
                                    beta1,
                                    beta2,
                                    learning_rate,
                                    epsilon)[["Velocity"]]
        s = update_params_with_adam(parameters,
                                    gradients,
                                    v,
                                    s,
                                    t,
                                    beta1,
                                    beta2,
                                    learning_rate,
                                    epsilon)[["S"]]
      }
    }
    
    if(verbose == T & (iter - 1) %% 10000 == 0){
      print(paste0("Cost after iteration " , iter - 1, ' = ', cost, sep = ' '))
    }
    
    if((iter - 1) %% 100 == 0){
      costs = c(costs,cost)
    }
  }
  
  if(output_layer_act != 'softmax'){
    pred_train <- predict_model(parameters,
                                X,
                                hidden_layer_act,
                                output_layer_act)
    Tr_acc <- mean(pred_train == Y) * 100
    pred_test <- predict_model(parameters,
                               X_test,
                               hidden_layer_act,
                               output_layer_act)
    Ts_acc <- mean(pred_test == Y_test) * 100
    cat(sprintf("Cost after iteration %d, = %05f;
Train Acc: %#.3f, Test Acc: %#.3f, \n",
                i, cost, Tr_acc, Ts_acc))
  }
  else if(output_layer_act != 'softmax'){
    pred_train <- predict_model(parameters,
                                X,
                                hidden_layer_act,
                                output_layer_act)
    Tr_acc <- mean((pred_train - 1) == Y)
    pred_test <- predict_model(parameters,
                               X_test,
                               hidden_layer_act,
                               output_layer_act)
    Ts_acc <- mean((pred_test - 1) == Y_test)
    cat(sprintf("Cost after iteration , %d, = %05f;
Train Acc: %#.3f, Test Acc: %#.3f, \n",
                i, cost, Tr_acc, Ts_acc))
  }
  
  end_time <- Sys.time()
  cat(sprintf("Application running time: %#.3f minutes", end_time - start_time ))
  
  return(list("parameters" = parameters, "costs" = costs))
}
