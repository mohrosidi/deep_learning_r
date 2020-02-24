# Import library and function
library(ggplot2)
library(gridExtra)
library(InspectChangepoint)
source("./src/Deep_Learning_with_R-Springer/2.Deep_Neural_Networks_1.R")

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

# Visualization
ggplot(data) + geom_point(aes(x = x,
                              y = y,
                              color = as.character(label)),
                          size = 1) +
  theme_bw(base_size = 15) +
  xlim(x_min, x_max) +
  ylim(y_min, y_max) +
  coord_fixed(ratio = 0.8) +
  theme(axis.ticks=element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text=element_blank(),
        axis.title=element_blank(),
        legend.position = 'none')

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

# Zero initialization
layers_dims <- c(2, 100, 1)

init_zero <- n_layer_model(t(trainX), 
                           trainY, 
                           t(testX), 
                           testY,
                           layers_dims, 
                           hidden_layer_act = "relu", 
                           output_layer_act = "sigmoid",
                           learning_rate = 0.03, 
                           num_iter = 5000, 
                           initialization = "zero",
                           print_cost = T)

# Decision boundary
step <- 0.01

x_min <- min(trainX[, 1]) - 0.2
x_max <- max(trainX[, 1]) + 0.2
y_min <- min(trainX[, 2]) - 0.2
y_max <- max(trainX[, 2]) + 0.2

grid <- as.matrix(expand.grid(seq(x_min, x_max, by = step), seq(y_min,
                  y_max, by = step)))

Z <- predict_model(init_zero$parameters, t(grid), hidden_layer_act = "relu",
                  output_layer_act = "sigmoid")

Z <- ifelse(Z == 0, 1, 2)

g1 <- ggplot() + geom_tile(aes(x = grid[, 1], y = grid[, 2],
                            fill = as.character(Z)), alpha = 0.3, show.legend = F) +
  geom_point(data = train_data, aes(x = x, y = y, color = as.character(trainY)),
           size = 1) + 
  theme_bw(base_size = 15) + coord_fixed(ratio = 0.8) +
  theme(axis.ticks = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.text = element_blank(),
        axis.title = element_blank(), legend.position = "none")

x_min <- min(testX[, 1]) - 0.2
x_max <- max(testX[, 1]) + 0.2
y_min <- min(testX[, 2]) - 0.2
y_max <- max(testX[, 2]) + 0.2

grid <- as.matrix(expand.grid(seq(x_min, x_max, by = step), seq(y_min,
                y_max, by = step)))

Z <- predict_model(init_zero$parameters, t(grid), hidden_layer_act = "relu",
                output_layer_act = "sigmoid")

Z <- ifelse(Z == 0, 1, 2)

g2 <- ggplot() + geom_tile(aes(x = grid[, 1], y = grid[, 2],
                        fill = as.character(Z)), alpha = 0.3, show.legend = F) +
  geom_point(data = test_data, aes(x = x, y = y, 
              color = as.character(testY)),size = 1) + 
  theme_bw(base_size = 15) + coord_fixed(ratio = 0.8) +
  theme(axis.ticks = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.text = element_blank(),
        axis.title = element_blank(), legend.position = "none")

grid.arrange(g1, g2, ncol = 2, nrow = 1)

# Random initialization
layers_dims <- c(2, 100, 1)

init_random <- n_layer_model(t(trainX),
                             trainY,
                             t(testX),
                             testY,
                             layers_dims,
                             hidden_layer_act = 'relu',
                             output_layer_act = 'sigmoid',
                             learning_rate = 0.03,
                             num_iter = 5000,
                             initialization = "random",
                             print_cost = T)

# Decision boundary
step <- 0.01

x_min <- min(trainX[, 1]) - 0.2
x_max <- max(trainX[, 1]) + 0.2
y_min <- min(trainX[, 2]) - 0.2
y_max <- max(trainX[, 2]) + 0.2

grid <- as.matrix(expand.grid(seq(x_min, x_max, by = step), seq(y_min,
                  y_max, by = step)))

Z <- predict_model(init_random$parameters, t(grid), hidden_layer_act = "relu",
                  output_layer_act = "sigmoid")

Z <- ifelse(Z == 0, 1, 2)

g1 <- ggplot() + geom_tile(aes(x = grid[, 1], y = grid[, 2],
                            fill = as.character(Z)), alpha = 0.3, show.legend = F) +
  geom_point(data = train_data, aes(x = x, y = y, color = as.character(trainY)),
           size = 1) + 
  theme_bw(base_size = 15) + coord_fixed(ratio = 0.8) +
  theme(axis.ticks = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.text = element_blank(),
        axis.title = element_blank(), legend.position = "none")

x_min <- min(testX[, 1]) - 0.2
x_max <- max(testX[, 1]) + 0.2
y_min <- min(testX[, 2]) - 0.2
y_max <- max(testX[, 2]) + 0.2

grid <- as.matrix(expand.grid(seq(x_min, x_max, by = step), seq(y_min,
                y_max, by = step)))

Z <- predict_model(init_random$parameters, t(grid), hidden_layer_act = "relu",
                output_layer_act = "sigmoid")

Z <- ifelse(Z == 0, 1, 2)

g2 <- ggplot() + geom_tile(aes(x = grid[, 1], y = grid[, 2],
                        fill = as.character(Z)), alpha = 0.3, show.legend = F) +
  geom_point(data = test_data, aes(x = x, y = y, 
              color = as.character(testY)),size = 1) + 
  theme_bw(base_size = 15) + coord_fixed(ratio = 0.8) +
  theme(axis.ticks = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.text = element_blank(),
        axis.title = element_blank(), legend.position = "none")

grid.arrange(g1, g2, ncol = 2, nrow = 1)

# Xavier initialization
layers_dims <- c(2, 100, 1)

init_Xavier <- n_layer_model(t(trainX),
                            trainY,
                            t(testX),
                            testY,
                            layers_dims,
                            hidden_layer_act = 'relu',
                            output_layer_act = 'sigmoid',
                            learning_rate = 0.03,
                            num_iter = 5000,
                            initialization = "Xavier",
                            print_cost = T)

# Decision boundary
step <- 0.01

x_min <- min(trainX[, 1]) - 0.2
x_max <- max(trainX[, 1]) + 0.2
y_min <- min(trainX[, 2]) - 0.2
y_max <- max(trainX[, 2]) + 0.2

grid <- as.matrix(expand.grid(seq(x_min, x_max, by = step), seq(y_min,
                  y_max, by = step)))

Z <- predict_model(init_Xavier$parameters, t(grid), hidden_layer_act = "relu",
                  output_layer_act = "sigmoid")

Z <- ifelse(Z == 0, 1, 2)

g1 <- ggplot() + geom_tile(aes(x = grid[, 1], y = grid[, 2],
                            fill = as.character(Z)), alpha = 0.3, show.legend = F) +
  geom_point(data = train_data, aes(x = x, y = y, color = as.character(trainY)),
           size = 1) + 
  theme_bw(base_size = 15) + coord_fixed(ratio = 0.8) +
  theme(axis.ticks = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.text = element_blank(),
        axis.title = element_blank(), legend.position = "none")

x_min <- min(testX[, 1]) - 0.2
x_max <- max(testX[, 1]) + 0.2
y_min <- min(testX[, 2]) - 0.2
y_max <- max(testX[, 2]) + 0.2

grid <- as.matrix(expand.grid(seq(x_min, x_max, by = step), seq(y_min,
                y_max, by = step)))

Z <- predict_model(init_Xavier$parameters, t(grid), hidden_layer_act = "relu",
                output_layer_act = "sigmoid")

Z <- ifelse(Z == 0, 1, 2)

g2 <- ggplot() + geom_tile(aes(x = grid[, 1], y = grid[, 2],
                        fill = as.character(Z)), alpha = 0.3, show.legend = F) +
  geom_point(data = test_data, aes(x = x, y = y, 
              color = as.character(testY)),size = 1) + 
  theme_bw(base_size = 15) + coord_fixed(ratio = 0.8) +
  theme(axis.ticks = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.text = element_blank(),
        axis.title = element_blank(), legend.position = "none")

grid.arrange(g1, g2, ncol = 2, nrow = 1)

# He initialization
layers_dims <- c(2, 100, 1)

init_He <- n_layer_model(t(trainX),
                        trainY,
                        t(testX),
                        testY,
                        layers_dims,
                        hidden_layer_act='relu',
                        output_layer_act = 'sigmoid',
                        learning_rate = 0.03,
                        num_iter = 5000,
                        initialization = "He",
                        print_cost = T)

# Decision boundary
step <- 0.01

x_min <- min(trainX[, 1]) - 0.2
x_max <- max(trainX[, 1]) + 0.2
y_min <- min(trainX[, 2]) - 0.2
y_max <- max(trainX[, 2]) + 0.2

grid <- as.matrix(expand.grid(seq(x_min, x_max, by = step), seq(y_min,
                  y_max, by = step)))

Z <- predict_model(init_He$parameters, t(grid), hidden_layer_act = "relu",
                  output_layer_act = "sigmoid")

Z <- ifelse(Z == 0, 1, 2)

g1 <- ggplot() + geom_tile(aes(x = grid[, 1], y = grid[, 2],
                            fill = as.character(Z)), alpha = 0.3, show.legend = F) +
  geom_point(data = train_data, aes(x = x, y = y, color = as.character(trainY)),
           size = 1) + 
  theme_bw(base_size = 15) + coord_fixed(ratio = 0.8) +
  theme(axis.ticks = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.text = element_blank(),
        axis.title = element_blank(), legend.position = "none")

x_min <- min(testX[, 1]) - 0.2
x_max <- max(testX[, 1]) + 0.2
y_min <- min(testX[, 2]) - 0.2
y_max <- max(testX[, 2]) + 0.2

grid <- as.matrix(expand.grid(seq(x_min, x_max, by = step), seq(y_min,
                y_max, by = step)))

Z <- predict_model(init_He$parameters, t(grid), hidden_layer_act = "relu",
                output_layer_act = "sigmoid")

Z <- ifelse(Z == 0, 1, 2)

g2 <- ggplot() + geom_tile(aes(x = grid[, 1], y = grid[, 2],
                        fill = as.character(Z)), alpha = 0.3, show.legend = F) +
  geom_point(data = test_data, aes(x = x, y = y, 
              color = as.character(testY)),size = 1) + 
  theme_bw(base_size = 15) + coord_fixed(ratio = 0.8) +
  theme(axis.ticks = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.text = element_blank(),
        axis.title = element_blank(), legend.position = "none")

grid.arrange(g1, g2, ncol = 2, nrow = 1)
