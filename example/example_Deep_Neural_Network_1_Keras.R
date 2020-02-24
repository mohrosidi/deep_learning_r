# Cat vs Dog Clasification

# Import package and function
library(EBImage)
library(pbapply)
library(tidyverse)
library(keras)

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
set.seed(123)
model <- keras_model_sequential()
model %>%
  layer_dense(units = 100, activation = 'relu', input_shape = c(12288)) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dense(units = 15, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  compile(
    optimizer = optimizer_adam(),
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )
summary(model)


