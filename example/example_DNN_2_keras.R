# Import library
library(keras)
library(magrittr)
library(ggplot2)
library(gridExtra)
library(InspectChangepoint)

# Data
N <- 400 # number of points per class
D <- 2 # dimensionality
K <- 2 # number of classes
X <- data.frame() # data matrix (each row = single example)
Y <- data.frame() # class labels
set.seed(308)
for (j in (1:2)) {
  r <- seq(0.05, 1, length.out = N) # radius
  t <- seq((j - 1) * 4.7, j * 4.7, length.out = N) + rnorm(N,
                                                           sd = 0.3) # theta
  Xtemp <- data.frame(x = r * sin(t), y = r * cos(t))
  ytemp <- data.frame(matrix(j, N, 1))
  X <- rbind(scale(X), Xtemp)
  Y <- rbind(Y, ytemp)
}
data <- cbind(X, Y)
colnames(data) <- c(colnames(X), "label")
x_min <- min(X[, 1]) - 0.2
x_max <- max(X[, 1]) + 0.2
y_min <- min(X[, 2]) - 0.2
y_max <- max(X[, 2]) + 0.2

# Data spliting
indexes <- sample(1:800, 600)
train_data <- data[indexes, ]
test_data <- data[-indexes, ]
trainX <- train_data[, c(1, 2)]
trainY <- train_data[, 3]
testX <- test_data[, c(1, 2)]
testY <- test_data[, 3]
trainY <- ifelse(trainY == 1, 0, 1)
testY <- ifelse(testY == 1, 0, 1)
dim(train_data)
dim(test_data)

scale.trainX <- scale(trainX)
scale.testX <- scale(testX)

# Model
set.seed(1)

model <- keras_model_sequential() %>%
  layer_dense(units = 400, activation = "relu", input_shape = 2) %>%
  layer_dense(units = 200, activation = "relu", input_shape = 2) %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  compile(
    optimizer = optimizer_adam(),
    loss = 'binary_crossentropy',
    metrics = 'accuracy'
  )

learn <- model %>% fit(scale.trainX, trainY,
                       epochs = 300,
                       batch_size = 64,
                       validation_split = 0.2,
                       verbose = FALSE)

learn

plot(learn)

# Model with Adjust Epoch
set.seed(1)

model <- keras_model_sequential() %>%
  layer_dense(units = 175, activation = "relu", input_shape = 2) %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  compile(
    optimizer = optimizer_adam(),
    loss = 'binary_crossentropy',
    metrics = 'accuracy'
  )

learn <- model %>% fit((scale.trainX), trainY,
                       epochs = 300,
                       batch_size = 64,
                       validation_split = 0.2,
                       verbose = FALSE,
                       callbacks = list(callback_early_stopping(patience = 100)))

learn

plot(learn)

# Model with Batch Normalization
set.seed(1)

model <- keras_model_sequential() %>%
  layer_dense(units = 175, activation = "relu", input_shape = 2) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  compile(
    optimizer = optimizer_adam(),
    loss = 'binary_crossentropy',
    metrics = 'accuracy'
  )

learn <- model %>% fit((scale.trainX), trainY,
                       epochs = 300,
                       batch_size = 64,
                       validation_split = 0.2,
                       verbose = FALSE,
                       callbacks = list(callback_early_stopping(patience = 100)))

learn

plot(learn)

# Model with Dropout
set.seed(1)

model <- keras_model_sequential() %>%
  layer_dense(units = 175, activation = "relu", input_shape = 2) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  compile(
    optimizer = optimizer_adam(),
    loss = 'binary_crossentropy',
    metrics = 'accuracy'
  )

learn <- model %>% fit((scale.trainX), trainY,
                       epochs = 300,
                       batch_size = 64,
                       validation_split = 0.2,
                       verbose = FALSE,
                       callbacks = list(callback_early_stopping(patience = 100)))

learn

plot(learn)

# Model with Weight Regularization
set.seed(1)

model <- keras_model_sequential() %>%
  layer_dense(units = 175, activation = "relu",
              input_shape = 2,
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  compile(
    optimizer = optimizer_adam(),
    loss = 'binary_crossentropy',
    metrics = 'accuracy'
  )

learn <- model %>% fit((scale.trainX), trainY,
                       epochs = 300,
                       batch_size = 64,
                       validation_split = 0.2,
                       verbose = FALSE,
                       callbacks = list(callback_early_stopping(patience = 100)))

learn

plot(learn)

# Model with Adjusted Learning Rate
set.seed(1)

model <- keras_model_sequential() %>%
  layer_dense(units = 175,
              activation = "relu", input_shape = 2,
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  compile(
    optimizer = optimizer_adam(),
    loss = 'binary_crossentropy',
    metrics = 'accuracy'
  )

learn <- model %>% fit((scale.trainX), trainY,
                       epochs = 300,
                       batch_size = 64,
                       validation_split = 0.2,
                       verbose = FALSE,
                       callbacks = list(callback_early_stopping(patience = 100),
                                        callback_reduce_lr_on_plateau()))

learn

plot(learn)


# Prediction
model %>% predict(scale.testX[1:10, ])
predictions <- model %>% evaluate(scale.testX, testY)

