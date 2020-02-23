# Cat vs Dog Clasification

# Import package and function
library(EBImage)
library(pbapply)
source("./src/Deep_Learning_with_R-Springer/1.simple_DNN.R")

# Define path for the data
file_path_train <- "./data/cat-vs-dog/train"
file_path_test <- "./data/cat-vs-dog/test"

# Standardize the size of images
height = 64
width = 64
channels = 3

# Reshape image function
extract_feature <- function(dir_path, width, height) {
  img_size <- width * height
  
  images <- list.files(dir_path)
  label <- ifelse(grepl("dog", images) == T, 1, 0)
  print(paste("Processing", length(images), "images"))
  
  feature_list <- pblapply(images, function(imgname) {
    
    img <- readImage(file.path(dir_path, imgname))
    img_resized <- EBImage::resize(img, w = width, h = height)
    img_matrix <- matrix(reticulate::array_reshape(img_resized, (width *
                                    height * channels)), nrow = width * height * channels)
    img_vector <- as.vector(t(img_matrix))
    
    return(img_vector)
  })
  
  feature_matrix <- do.call(rbind, feature_list)
  
  return(list(t(feature_matrix), label))
}

# Reshape the images
data_train <- extract_feature(file_path_train, width, height)
data_test <- extract_feature(file_path_test, width, height)

# Data preprocessing
trainx <-data_train[[1]]
trainy <-data_train[[2]]
dim(trainx)

testx <-data_test[[1]]
testy<- data_test[[2]]
dim(testx)

# Data visualization
par(mfrow = c(1, 2))

images <- list.files(file_path_train)
img <- readImage(file.path(file_path_train, images[101]))
EBImage::display(img, method = "raster")
EBImage::display(matrix(as.matrix(trainx[, 101]),
                        c(64, 64, 3),
                        byrow = TRUE),
                 method = "raster")

images <- list.files(file_path_test)
img <- readImage(file.path(file_path_test, images[18]))
EBImage::display(img, method = "raster")
EBImage::display(matrix(as.matrix(testx[, 18]),
                        c(64, 64, 3),
                        byrow = TRUE),
                 method = "raster")

par(mfrow = c(1, 1))

# Data normalization
trainx <- scale(trainx)
testx <- scale(testx)

# Model
model <- simple_model(trainx,
                     trainy,
                     testx,
                     testy,
                     num_iter = 5000,
                     learning_rate = 0.01,
                     print_cost = TRUE)

# Plot Cost vs Iteration
x <- c(1:5000)
y <- model$costs
smoothingSpline = smooth.spline(x, y, spar = 0.35)

plot(NULL, type = "n",
     xlab = "Iterations", ylab = "Cost",
     xlim = c(1, 5000), ylim = c(0, 1),
     xaxt = "n", yaxt = "n",
     cex.lab = 0.7)

lines(smoothingSpline, col = "deepskyblue4")
axis(side = 1, col = "black", cex.axis = 0.7)
axis(side = 2, col = "black", cex.axis = 0.7)

legend(1550, 0.9, inset = 0.001, c("Learning rate = 0.01"), cex = 0.6)

# Tuning learning rate hyperparameter
learning_rates <- c(0.01, 0.002, 0.005)
models <- list()
smoothingSpline <- list()

plot(NULL, type = "n",
     xlab = "Iterations", ylab = "Cost",
     xlim = c(1, 5000), ylim = c(0, 1),
     xaxt = "n", yaxt = "n",
     cex.lab = 0.7)

for(i in 1:length(learning_rates)){
  cat(sprintf("Learning rate: %#.3f \n", learning_rates[i]))
  models[[i]] = simple_model(trainx,
                             trainy,
                             testx,
                             testy,
                             num_iter = 5000,
                             learning_rate = learning_rates[i],
                             print_cost = F)
  
  cat('\n-------------------------------------------------------\n')
  
  x <- c(1:5000)
  y <- unlist(models[[i]]$costs)
  smoothingSpline = smooth.spline(x, y, spar = 0.35)
  
  lines(smoothingSpline, col = i + 2, lwd = 2)
}

axis(side = 1, col = "black", cex.axis = 0.7)
axis(side = 2, col = "black", cex.axis = 0.7)

legend("topright", inset = 0.001,
       c("Learning rate = 0.01",
          "Learning rate = 0.002",
         "Learning rate = 0.005"),
       lwd = c(2, 2, 2),
       lty = c(1, 1, 1),
       col = c("green3", "blue", "cyan"),
       cex = 0.6)
