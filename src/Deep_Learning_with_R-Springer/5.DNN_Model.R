# Import function
source("./src/Deep_Learning_with_R-Springer/4.DNN_Optimization_using_regularization.R")

# DNN function
DNN_model <- function(X,
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
                      keep_prob,
                      lambd,
                      print_cost = F){
  
  start_time <- Sys.time()
  costs <- NULL
  converged = FALSE
  param <- NULL
  t = 0
  iter = 0
  set.seed = 1
  seed = 10
  num_classes = length(unique(Y))
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
        AL = forward_prop(mini_batch_X, parameters, hidden_layer_act,
                          output_layer_act)[['AL']]
        caches = forward_prop(mini_batch_X, parameters, hidden_layer_act,
                              output_layer_act)[['caches']]
      }
      else if(keep_prob < 1){
        AL = forward_prop_Reg(mini_batch_X, parameters, hidden_layer_act,
                              output_layer_act, keep_prob)[['AL']]
        caches = forward_prop_Reg(mini_batch_X, parameters,
                                  hidden_layer_act,
                                  output_layer_act,
                                  keep_prob)[['caches']]
        dropout_matrix = forward_prop_Reg(mini_batch_X, parameters,
                                          hidden_layer_act,
                                          output_layer_act,
                                          keep_prob)[['dropout_matrix']]
      }
      
      cost <- compute_cost_with_Reg(AL, mini_batch_X, mini_batch_Y,
                                    num_classes,
                                    parameters,
                                    lambd,
                                    output_layer_act)
      
      # Backward propagation
      if(lambd == 0 & keep_prob == 1){
        gradients = back_prop(AL, mini_batch_Y, caches,
                              hidden_layer_act, output_layer_act)
      }
      else if(lambd != 0 & keep_prob == 1){
        gradients = back_prop_Reg(AL, mini_batch_X, mini_batch_Y,
                                  num_classes,
                                  caches, hidden_layer_act,
                                  output_layer_act, keep_prob = 1,
                                  dropout_matrix, lambd)
      }
      else if(lambd == 0 & keep_prob < 1){
        gradients = back_prop_Reg(AL, mini_batch_X, mini_batch_Y,
                                  num_classes, caches,
                                  hidden_layer_act,
                                  output_layer_act, keep_prob,
                                  dropout_matrix, lambd = 0)
      }
      
      if(optimizer == 'gd'){
        parameters = update_params(parameters, gradients, learning_rate)
      }
      else if(optimizer == 'momentum'){
        parameters = update_params_with_momentum(parameters, gradients, velocity,
                                                 beta,
                                                 learning_rate)[["parameters"]]
        velocity = update_params_with_momentum(parameters, gradients, velocity,
                                               beta,
                                               learning_rate)[["Velocity"]]
      }
      else if(optimizer == 'adam'){
        t = t + 1
        parameters = update_params_with_adam(parameters, gradients, v, s, t,
                                             beta1, beta2,
                                             learning_rate,
                                             epsilon)[["parameters"]]
        v = update_params_with_adam(parameters, gradients, v, s, t,
                                    beta1, beta2,
                                    learning_rate,
                                    epsilon)[["Velocity"]]
        s = update_params_with_adam(parameters, gradients, v, s, t,
                                    beta1, beta2,
                                    learning_rate,
                                    epsilon)[["S"]]
      }
    }
    
    costs <- append(costs, list(cost))
    
    if(print_cost == T & i %% 1000 == 0){
      cat(sprintf("Cost after epoch %d = %05f\n", i, cost))
    }
  }
  
  if(output_layer_act != 'softmax'){
    pred_train <- predict_model(parameters, X,
                                hidden_layer_act,
                                output_layer_act)
    Tr_acc <- mean(pred_train == Y) * 100
    pred_test <- predict_model(parameters, X_test,
                               hidden_layer_act,
                               output_layer_act)
    Ts_acc <- mean(pred_test == Y_test) * 100
    cat(sprintf("Cost after epoch %d, = %05f;
Train Acc: %#.3f, Test Acc: %#.3f, \n",
                i, cost, Tr_acc, Ts_acc))
  }
  else if(output_layer_act == 'softmax'){
    pred_train <- predict_model(parameters, X,
                                hidden_layer_act, output_layer_act)
    Tr_acc <- mean((pred_train - 1) == Y)
    pred_test <- predict_model(parameters, X_test,
                               hidden_layer_act, output_layer_act)
    Ts_acc <- mean((pred_test - 1) == Y_test)
    cat(sprintf("Cost after epoch , %d, = %05f;
Train Acc: %#.3f, Test Acc: %#.3f, \n",
                i, cost, Tr_acc, Ts_acc))
  }
  
  end_time <- Sys.time()
  cat(sprintf("Application running time: %#.3f seconds\n",
              end_time - start_time ))
  
  return(list("parameters" = parameters, "costs" = costs))
}