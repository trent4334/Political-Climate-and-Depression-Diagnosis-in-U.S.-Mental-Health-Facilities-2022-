###Final Version

#ML_data

#setwd("/Users/trentyu/Desktop/Messy Data Machine Learning/Final Project/data")
ML_data <- read.csv("ML_data")

drop_features <- c("NUMMHS", "ANXIETYFLG","MH1")
ML_data <- ML_data[, !(colnames(ML_data) %in% drop_features)]


# Ensure the target variable is a factor
ML_data$DEPRESSFLG <- as.factor(ML_data$DEPRESSFLG)

# Split data into training and testing sets
set.seed(42)
train_index <- createDataPartition(ML_data$DEPRESSFLG, p = 0.8, list = FALSE)
train_data <- ML_data[train_index, ]
test_data <- ML_data[-train_index, ]

# Train a Random Forest model
set.seed(42)
rf_model <- randomForest(DEPRESSFLG ~ ., data = train_data, importance = TRUE, ntree = 100)

# Print model summary
print(rf_model)

# Predict on the test set
test_predictions <- predict(rf_model, test_data, type = "response")
test_probabilities <- predict(rf_model, test_data, type = "prob")[, 2]

# Evaluate the model
conf_matrix <- confusionMatrix(test_predictions, test_data$DEPRESSFLG)
print(conf_matrix)

# Calculate AUC
pred <- prediction(test_probabilities, test_data$DEPRESSFLG)
auc <- performance(pred, "auc")@y.values[[1]]
cat("AUC: ", auc, "\n")







####--------------------------Comparison RF with other models----------------------------------############


ML_data$DEPRESSFLG <- as.factor(ML_data$DEPRESSFLG)

# Split data into train and test sets
set.seed(123)
trainIndex <- createDataPartition(ML_data$DEPRESSFLG, p = 0.8, list = FALSE)
trainData <- ML_data[trainIndex, ]
testData <- ML_data[-trainIndex, ]

set.seed(123)
rf_model <- randomForest(DEPRESSFLG ~ ., data = trainData, probability = TRUE)
rf_pred <- predict(rf_model, testData, type = "prob")[, 2]

rf_roc <- roc(testData$DEPRESSFLG, rf_pred)
rf_auc <- auc(rf_roc)
print(paste("Random Forest AUC:", rf_auc))


# Random Forest with limited tree depth
set.seed(123)
rf_model_limited <- randomForest(
  DEPRESSFLG ~ ., 
  data = trainData, 
  ntree = 100,          # Set the number of trees
  maxnodes = 20,        # Limit the maximum number of terminal nodes
  nodesize = 10          # Minimum number of observations per leaf node
)

# Predict probabilities on the test set
rf_pred_limited <- predict(rf_model_limited, testData, type = "prob")[, 2]

# Evaluate AUC
library(pROC)
rf_roc_limited <- roc(testData$DEPRESSFLG, rf_pred_limited)
rf_auc_limited <- auc(rf_roc_limited)
print(paste("Random Forest with limited tree depth AUC:", rf_auc_limited))

# AUC for Random Forest
library(pROC)
rf_roc <- roc(testData$DEPRESSFLG, rf_pred)
rf_auc <- auc(rf_roc)
print(paste("Random Forest AUC:", rf_auc))


#Naive

# Install and load the e1071 package
if (!require("e1071")) install.packages("e1071")
library(e1071)

# Ensure target variable is a factor
trainData$DEPRESSFLG <- as.factor(trainData$DEPRESSFLG)
testData$DEPRESSFLG <- as.factor(testData$DEPRESSFLG)

# Train Naive Bayes model
set.seed(123)
nb_model <- naiveBayes(DEPRESSFLG ~ ., data = trainData)

# Predict probabilities on the test set
nb_pred <- predict(nb_model, testData, type = "raw")[, 2]  # Get probabilities for class 1

# Calculate AUC
library(pROC)
nb_roc <- roc(testData$DEPRESSFLG, nb_pred)
nb_auc <- auc(nb_roc)
print(paste("Naive Bayes AUC:", nb_auc))


# Gradient Boosting Machine (GBM)
library(gbm)
trainData$DEPRESSFLG <- as.numeric(as.character(trainData$DEPRESSFLG)) # Ensure numeric
testData$DEPRESSFLG <- as.numeric(as.character(testData$DEPRESSFLG)) # Ensure numeric

set.seed(123)
gbm_model <- gbm(
  DEPRESSFLG ~ ., 
  data = trainData, 
  distribution = "bernoulli", 
  n.trees = 100, 
  interaction.depth = 3, 
  shrinkage = 0.01, 
  cv.folds = 5, 
  verbose = FALSE
)

# Predict probabilities and calculate AUC
gbm_pred <- predict(gbm_model, testData, n.trees = 100, type = "response")
gbm_roc <- roc(testData$DEPRESSFLG, gbm_pred)
gbm_auc <- auc(gbm_roc)
print(paste("GBM AUC:", gbm_auc))

# Logistic Regression
set.seed(123)

# Train the logistic regression model
logistic_model <- glm(DEPRESSFLG ~ ., data = trainData, family = binomial)

# Predict probabilities on the test set
logistic_pred <- predict(logistic_model, testData, type = "response")

# Calculate AUC
logistic_roc <- roc(testData$DEPRESSFLG, logistic_pred)
logistic_auc <- auc(logistic_roc)
print(paste("Logistic Regression AUC:", logistic_auc))



#Random Forest ROC curve
plot(rf_roc, col = "blue", main = "ROC Curve Comparison", lwd = 2)

#GBM ROC curve
lines(gbm_roc, col = "green", lwd = 2)

#Naive Bayes ROC curve
lines(nb_roc, col = "red", lwd = 2)

#Logistic Regression ROC curve
lines(logistic_roc, col = "purple", lwd = 2)

# Add legend
legend("bottomright", legend = c(
  paste("Random Forest (AUC:", round(rf_auc, 3), ")"),
  paste("GBM (AUC:", round(gbm_auc, 3), ")"),
  paste("Naive Bayes (AUC:", round(nb_auc, 3), ")"),
  paste("Logistic Regression (AUC:", round(logistic_auc, 3), ")")
), col = c("blue", "green", "red", "purple"), lwd = 2)

# Add text annotations for AUC scores
text(x = 0.6, y = 0.35, paste("Random Forest AUC:", round(rf_auc, 3)), col = "blue", cex = 0.9)
text(x = 0.6, y = 0.3, paste("GBM AUC:", round(gbm_auc, 3)), col = "green", cex = 0.9)
text(x = 0.6, y = 0.25, paste("Naive Bayes AUC:", round(nb_auc, 3)), col = "red", cex = 0.9)
text(x = 0.6, y = 0.2, paste("Logistic Regression AUC:", round(logistic_auc, 3)), col = "purple", cex = 0.9)





### Precision and Recall Graph
if (!require("PRROC")) install.packages("PRROC")
library(PRROC)

# Logistic Regression Precision-Recall Curve
logistic_pr <- pr.curve(
  scores.class0 = logistic_pred, 
  weights.class0 = as.numeric(testData$DEPRESSFLG), 
  curve = TRUE
)

# Random Forest Precision-Recall Curve
if (!require("PRROC")) install.packages("PRROC")
library(PRROC)

# Logistic Regression Precision-Recall Curve
logistic_pr <- pr.curve(
  scores.class0 = logistic_pred, 
  weights.class0 = as.numeric(testData$DEPRESSFLG), 
  curve = TRUE
)

# Random Forest Precision-Recall Curve
rf_pr <- pr.curve(
  scores.class0 = rf_pred, 
  weights.class0 = as.numeric(testData$DEPRESSFLG), 
  curve = TRUE
)

#Precision-Recall Curves
plot(logistic_pr, col = "blue", lwd = 2, main = "Precision-Recall Curve Comparison", legend = FALSE)
lines(rf_pr$curve[, 1], rf_pr$curve[, 2], col = "green", lwd = 2)

# Add Legend
legend("bottomright", legend = c("Logistic Regression", "Random Forest"), col = c("blue", "green"), lwd = 2)


#Confusion Matrices

# Threshold (Cutoff)
threshold <- 0.3 

# Logistic Regression Confusion Matrix
logistic_pred_class <- ifelse(logistic_pred > threshold, 1, 0)
logistic_cm <- table(Predicted = logistic_pred_class, Actual = testData$DEPRESSFLG)
logistic_accuracy <- sum(diag(logistic_cm)) / sum(logistic_cm)  # Accuracy calculation
print("Logistic Regression Confusion Matrix:")
print(logistic_cm)
print(paste("Logistic Regression Accuracy:", round(logistic_accuracy, 3)))

# Random Forest Confusion Matrix
rf_pred_class <- ifelse(rf_pred > threshold, 1, 0)
rf_cm <- table(Predicted = rf_pred_class, Actual = testData$DEPRESSFLG)
rf_accuracy <- sum(diag(rf_cm)) / sum(rf_cm)  # Accuracy calculation
print("Random Forest Confusion Matrix:")
print(rf_cm)
print(paste("Random Forest Accuracy:", round(rf_accuracy, 3)))


# Feature Importance
importance <- importance(rf_model)
importance_df <- data.frame(Feature = rownames(importance), Importance = importance[, 1])
importance_df <- importance_df[order(-importance_df$Importance), ]

cat("Top 20 Important Features:\n")
print(head(importance_df, 20))

library(ggplot2)
ggplot(head(importance_df, 20), aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top 20 Important Features for Depression Diagnosis",
       x = "Feature",
       y = "Importance") +
  theme_minimal()