# import library
library(pbapply)

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