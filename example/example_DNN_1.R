# Cat vs Dog Clasification

# Import package and function
library(EBImage)
library(pbapply)
source("./src/Deep_Learning_with_R-Springer/2.Deep_Neural_Networks_1.R")

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
layers_dims = c(12288, 3, 1)

two_layer_model = n_layer_model(trainx,
                                trainy,
                                testx,
                                testy,
                                layers_dims,
                                hidden_layer_act = 'tanh',
                                output_layer_act = 'sigmoid',
                                learning_rate = 0.01,
                                num_iter = 1500,
                                initialization = "random",
                                print_cost = T)


layers_dims = c(12288, 50, 30, 1)

three_layer_model = n_layer_model(trainx,
                                  trainy,
                                  testx,
                                  testy,
                                  layers_dims,
                                  hidden_layer_act = c('relu', 'relu'),
                                  output_layer_act = 'sigmoid',
                                  learning_rate = 0.045,
                                  num_iter = 1500,
                                  initialization = "random",
                                  print_cost = T)


layers_dims = c(nrow(trainx), 50, 20, 7, 2)

four_layer_model = n_layer_model(trainx,
                                 trainy,
                                 testx,
                                 testy,
                                 layers_dims,
                                 hidden_layer_act = c('relu', 'relu', 'tanh'),
                                 output_layer_act = 'softmax',
                                 learning_rate = 0.15,
                                 num_iter = 1500,
                                 initialization = "random",
                                 print_cost = T)

# Compute probabilities
compute_Proba <- function(parameters,
                          test_X,
                          hidden_layer_act,
                          output_layer_act){
  
  score <- forward_prop(test_X,
                        parameters,
                        hidden_layer_act,
                        output_layer_act)[['AL']]
  Probs <- list(round(score * 100, 2))
  return (Probs)
}

Prob <- compute_Proba(two_layer_model$parameters,
                      testx,
                      hidden_layer_act = c('relu', 'relu'),
                      output_layer_act = 'sigmoid')

labels = ifelse(testy == 1, "dog", "cat")

predicted <- ifelse(
  predict_model(two_layer_model$parameters,
                testx,
                hidden_layer_act = c('relu', 'relu'),
                output_layer_act = 'sigmoid') == 0, 'cat', 'dog')

error <- ifelse(predicted == labels, 'No', 'Yes')

index <- c(1:length(labels))

Probs <- as.vector(unlist(Prob[index]))

par(mfrow = c(5, 10), mar = rep(0, 4))

for(i in 1:length(index)){
  image(t(apply(matrix(as.matrix(testx[, index[i]]),
                       c(64, 64, 3),
                       byrow = TRUE), 1, rev)),
        method = 'raster',
        col = gray.colors(12),
        axes = F)
  legend("topright", legend = predicted[i],
         text.col = ifelse(error[i] == 'Yes', 2, 4),
         bty = "n",
         text.font = 1.5)
  legend("bottomright", legend = Probs[i], bty = "n", col = "white")
}
