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

# Regular Poisson GLM
model_poisson <- glm(Condition ~ ., family = poisson, data = train_data)

# Transformed Linear Model
mod_transf <- lm(I((Condition + 0.1)^2) ~ . + Year:Material, data = train_data)

# Prepare the data - glmnet requires matrix format
X_train <- model.matrix(Condition ~ ., data = train_data)[, -1]  # Remove intercept column
y_train <- train_data$Condition
X_test <- model.matrix(Condition ~ ., data = test_data)[, -1]
y_test <- test_data$Condition

# Fit Poisson LASSO (alpha = 1 for LASSO, alpha = 0 for Ridge)
cv_lasso_poisson <- cv.glmnet(
  x = X_train,
  y = y_train,
  family = "poisson", 
  alpha = 1,           
  nfolds = 10         
)

# Plot cross-validation curve
plot(cv_lasso_poisson)

# Best lambda values
lambda_min <- cv_lasso_poisson$lambda.min      # Minimum CV error
lambda_1se <- cv_lasso_poisson$lambda.1se      # 1 SE rule (more parsimonious)

# Fit final model with best lambda
model_lasso_poisson <- glmnet(
  x = X_train,
  y = y_train,
  family = "poisson",
  alpha = 1,
  lambda = lambda_1se  # Or use lambda_min
)

# View selected variables (non-zero coefficients)
coef(model_lasso_poisson)

# Ridge Regression (alpha = 0)
cv_ridge_poisson <- cv.glmnet(
  x = X_train,
  y = y_train,
  family = "poisson",
  alpha = 0,           # Ridge penalty
  nfolds = 10
)

# Plot CV curve
plot(cv_ridge_poisson)

# Fit with best lambda
model_ridge_poisson <- glmnet(
  x = X_train,
  y = y_train,
  family = "poisson",
  alpha = 0,
  lambda = cv_ridge_poisson$lambda.1se
)


# Regular Poisson - Training
pred_train_poisson <- predict(model_poisson, type = "response")
rmse_train_poisson <- sqrt(mean((train_data$Condition - pred_train_poisson)^2))

# Regular Poisson - Test
pred_test_poisson <- predict(model_poisson, newdata = test_data, type = "response")
rmse_test_poisson <- sqrt(mean((test_data$Condition - pred_test_poisson)^2))

# LASSO Poisson - Training
pred_train_lasso <- predict(model_lasso_poisson, newx = X_train, type = "response")
rmse_train_lasso <- sqrt(mean((y_train - pred_train_lasso)^2))

# LASSO Poisson - Test
pred_test_lasso <- predict(model_lasso_poisson, newx = X_test, type = "response")
rmse_test_lasso <- sqrt(mean((y_test - pred_test_lasso)^2))

# Ridge Poisson - Training
pred_train_ridge <- predict(model_ridge_poisson, newx = X_train, type = "response")
rmse_train_ridge <- sqrt(mean((y_train - pred_train_ridge)^2))

# Ridge Poisson - Test
pred_test_ridge <- predict(model_ridge_poisson, newx = X_test, type = "response")
rmse_test_ridge <- sqrt(mean((y_test - pred_test_ridge)^2))

# Transformed Model - Training
pred_train_transf_raw <- predict(mod_transf, newdata = train_data)
pred_train_transf <- sqrt(pmax(pred_train_transf_raw, 0)) - 0.1  
rmse_train_transf <- sqrt(mean((train_data$Condition - pred_train_transf)^2))

# Transformed Model - Test
pred_test_transf_raw <- predict(mod_transf, newdata = test_data)
pred_test_transf <- sqrt(pmax(pred_test_transf_raw, 0)) - 0.1
rmse_test_transf <- sqrt(mean((test_data$Condition - pred_test_transf)^2))


comparison <- data.frame(
  Model = c("Poisson", "LASSO Poisson", "RIDGE Poisson", "Transformed LM"),
  Train_RMSE = c(rmse_train_poisson, rmse_train_lasso, rmse_train_ridge, rmse_train_transf),
  Test_RMSE = c(rmse_test_poisson, rmse_test_lasso, rmse_test_ridge, rmse_test_transf),
  Difference = c(rmse_test_poisson - rmse_train_poisson, 
                 rmse_test_lasso - rmse_train_lasso,
                 rmse_test_ridge - rmse_train_ridge,
                 rmse_test_transf - rmse_train_transf)
)

print(comparison)
