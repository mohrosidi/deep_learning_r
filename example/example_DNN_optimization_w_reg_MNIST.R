# Import function
source("./src/Deep_Learning_with_R-Springer/5.DNN_Model.R")


# Data
data <- dslabs::read_mnist()

train <- t(data.matrix(data$train$images)[1:5000,]/256)
ytrain <- data$train$labels[1:5000]

test <- t(data.matrix(data$test$images)[1:1000]/256)
ytest <- data$test$labels[1:1000]

# Model
model <- DNN_model(train,
                 ytrain,
                 test,
                 ytest,
                 layers_dims = c(nrow(train), 30, 10, 10),
                 hidden_layer_act = c('relu', 'relu', 'relu'),
                 output_layer_act = 'softmax',
                 optimizer = 'adam',
                 learning_rate = 0.001,
                 mini_batch_size = 32,
                 num_epochs = 300,
                 initialization = 'He',
                 beta = 0.9,
                 beta1 = 0.9,
                 beta2 = 0.999,
                 epsilon = 1e-8,
                 keep_prob = 1,
                 lambd = 0.0001,
                 print_cost = T)
