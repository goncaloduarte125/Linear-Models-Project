# Load required libraries
library(ggplot2)
library(dplyr)
library(MASS)
library(glmnet)

# Set seed for reproducibility
set.seed(3185)

# ===========================
# 1. DATA PREPARATION
# ===========================

# Remove Structure_id from the dataset
modeling_data <- group_03 %>% 
  dplyr::select(-Structure_id)


# Get number of observations
n <- nrow(modeling_data)
n_train <- floor(0.8 * n)

# Randomly sample indices for training set
train_indices <- sample(1:n, size = n_train, replace = FALSE)

# Split data
train_data <- modeling_data[train_indices, ]
test_data <- modeling_data[-train_indices, ]

cat("\nTraining set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

# ===========================
# 2. MODEL FITTING
# ===========================


model_poisson <- glm(Condition ~ . - Structure_id + Year:Material, 
                     family = poisson, 
                     data = db_train) 


mod_poi_step <- step(model_poisson, direction = "both", trace = 0)
summary(mod_poi_step)


anova(mod_poi_step, model_poisson, test = "Chisq")


pred_poi_test <- predict(mod_poi_step, newdata = db_test, type = "response")


actuals <- db_test$Condition
rmse_poi <- sqrt(mean((actuals - pred_poi_test)^2))

print(paste("RMSE Poisson:", round(rmse_poi, 4)))