# All the models loaded can be accessed here:
#https://drive.google.com/drive/u/0/folders/11E0_rjPgh2udB0IboIvCDqFSMG4wDaYU
# An example script of model tuning and a complete script for fitting models are 
# available in the folder as well. 
set.seed(42)

library(tidymodels)
library(xgboost)
library(yardstick)
library(dials)
library(recipes)
library(stacks)
library(readr)
library(bonsai)
library(lightgbm)
library(kknn)
library(randomForest)
library(kernlab)

# load train set
train_data <- read.csv('train.csv')
train_data = train_data %>% mutate(action_taken = as.factor(action_taken))
test_data <- read.csv('test.csv')
xgb_fit <- readRDS("xgb_fit.rds")
xgb_fit_21 <- readRDS("xgb_fit_21.rds")
rf_fit <- readRDS("rf_fit.rds")
rf_fit_21 <- readRDS("rf_fit_21.rds")
lightgbm_fit <- readRDS("lightgbm_fit.rds")
lightgbm_fit$fit$fit$fit <- readRDS.lgb.Booster("lgbm_booster.rds")
lightgbm_fit_21 <- readRDS("lightgbm_fit_21.rds")
lightgbm_fit_21$fit$fit$fit <- readRDS.lgb.Booster("lgbm_booster_21.rds")
en_fit <- readRDS("en_fit.rds")
en_fit_21 <- readRDS("en_fit_21.rds")
en_fit_a <- readRDS("en_fit_a.rds")

train_data$pred_xgb <- predict(xgb_fit, new_data=train_data)
train_data$pred_xgb_21 <- predict(xgb_fit_21, new_data=train_data)
train_data$pred_rf <- predict(rf_fit, new_data=train_data)
train_data$pred_rf_21 <- predict(rf_fit_21, new_data=train_data)
train_data$pred_lightgbm <- predict(lightgbm_fit, new_data=train_data)
train_data$pred_lightgbm_21 <- predict(lightgbm_fit_21, new_data=train_data)
train_data$pred_en <- predict(en_fit, new_data=train_data)
train_data$pred_en_21 <- predict(en_fit_21, new_data=train_data)
train_data$pred_en_a <- predict(en_fit_a, new_data=train_data)

# Create a new data frame to hold the predictions and the actual target
train_meta_data <- data.frame(
  pred_xgb = train_data$pred_xgb,
  pred_xgb_21 = train_data$pred_xgb_21,
  pred_rf = train_data$pred_rf,
  pred_rf_21 = train_data$pred_rf_21,
  #pred_knn = train_data$pred_knn,
  #pred_knn_21 = train_data$pred_knn_21,
  pred_lightgbm = train_data$pred_lightgbm,
  pred_lightgbm_21 = train_data$pred_lightgbm_21,
  pred_en = train_data$pred_en,
  pred_en_21 = train_data$pred_en_21,
  pred_en_a = train_data$pred_en_a,
  target = train_data$action_taken
)

train_meta_data = train_meta_data%>%
  group_by(target) %>%
  sample_frac(0.4) # memory limit of stacked model

# Train a meta-model
meta_model <- randomForest(target ~ ., data=train_meta_data)

test_data$pred_xgb <- predict(xgb_fit, new_data=test_data)
test_data$pred_xgb_21 <- predict(xgb_fit_21, new_data=test_data)
test_data$pred_rf <- predict(rf_fit, new_data=test_data)
test_data$pred_rf_21 <- predict(rf_fit_21, new_data=test_data)
test_data$pred_lightgbm <- predict(lightgbm_fit, new_data=test_data)
test_data$pred_lightgbm_21 <- predict(lightgbm_fit_21, new_data=test_data)
test_data$pred_en <- predict(en_fit, new_data=test_data)
test_data$pred_en_21 <- predict(en_fit_21, new_data=test_data)
test_data$pred_en_a <- predict(en_fit_a, new_data=test_data)

# Create a new data frame to hold the test predictions
test_meta_data <- data.frame(
  pred_xgb = test_data$pred_xgb,
  pred_xgb_21 = test_data$pred_xgb_21,
  pred_rf = test_data$pred_rf,
  pred_rf_21 = test_data$pred_rf_21,
  pred_lightgbm = test_data$pred_lightgbm,
  pred_lightgbm_21 = test_data$pred_lightgbm_21,
  pred_en = test_data$pred_en,
  pred_en_21 = test_data$pred_en_21,
  pred_en_a = test_data$pred_en_a
)

# Generate final predictions using the meta-model
final_predictions <- predict(meta_model, newdata=test_meta_data)
result_df <- bind_cols(test_data %>% select(id), .pred_class = final_predictions)
result_df <- result_df %>% rename(action_taken = .pred_class)
write.csv(result_df, "team_2_stack_3.csv", row.names = FALSE)