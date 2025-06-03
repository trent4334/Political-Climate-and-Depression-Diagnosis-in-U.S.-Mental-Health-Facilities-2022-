setwd("/Users/trentyu/Desktop/Messy Data Machine Learning/Final Project/data")

library(dplyr)  
library(tidyr)  
library(ggplot2)   
library(caret)     
library(pROC)      
library(randomForest)   

dataset <- read.csv("data.csv")

#Political Stance

red_states <- c(1, 2, 5, 12, 16, 18, 19, 20, 21, 22, 28, 29, 30, 31, 37, 38, 39, 40, 45, 46, 47, 48, 49, 54, 56)
blue_states <- c(4, 6, 8, 9, 10, 11, 13, 15, 17, 23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 41, 42, 44, 50, 51, 53, 55)

# Add the "Political_Stance" column to your dataset
dataset$Political_Stance <- ifelse(dataset$STATEFIP %in% red_states, "Republican",
                                   ifelse(dataset$STATEFIP %in% blue_states, "Democratic", NA))

table(dataset$Political_Stance)

colnames(dataset)



# Select important variables
selected_vars <- dataset %>%
  select(Political_Stance, DEPRESSFLG, AGE, GENDER, MARSTAT, EMPLOY, EDUC,ETHNIC)

# Convert categorical variables to dummy variables
library(fastDummies)

dummy_vars <- dummy_cols(selected_vars, 
                         select_columns = c("Political_Stance", "GENDER", "MARSTAT", "EMPLOY", "EDUC","AGE","ETHNIC"), 
                         #remove_first_dummy = TRUE, 
                         remove_selected_columns = TRUE)

 dummy_vars$DEPRESSFLG <- selected_vars$DEPRESSFLG


dummy_vars$Children <- ifelse(dummy_vars$AGE_1 == 1 | dummy_vars$AGE_2 == 1 | dummy_vars$AGE_3 == 1, 1, 0)
dummy_vars$Young_Adult <- ifelse(dummy_vars$AGE_4 == 1 | dummy_vars$AGE_5 == 1, 1, 0)
dummy_vars$Adult <- ifelse(dummy_vars$AGE_6 == 1 | dummy_vars$AGE_7 == 1 | dummy_vars$AGE_8 == 1 | dummy_vars$AGE_9 == 1, 1, 0)
dummy_vars$Middle_Aged_Adults <- ifelse(dummy_vars$AGE_10 == 1 | dummy_vars$AGE_11 == 1 | dummy_vars$AGE_12 == 1 | dummy_vars$AGE_13 == 1, 1, 0)
dummy_vars$Senior <- ifelse(dummy_vars$AGE_14 == 1, 1, 0)

dummy_vars$Hispanic <- ifelse(dummy_vars$ETHNIC_1 == 1 | dummy_vars$ETHNIC_2 == 1 | dummy_vars$ETHNIC_3 == 1, 1, 0)
dummy_vars$Non_Hispanic <- ifelse(dummy_vars$ETHNIC_4 == 1, 1, 0)

column_mapping <- c(
  "EMPLOY_1" = "EMPLOY_Full_time",
  "EMPLOY_2" = "EMPLOY_Part_time",
  "EMPLOY_3" = "EMPLOY_Employed",
  "EMPLOY_4" = "EMPLOY_Unemployed",
  "EMPLOY_5" = "EMPLOY_Not_in_labor_force",
  "EDUC_1" = "EDUC_Special_education",
  "EDUC_2" = "EDUC_0to8",
  "EDUC_3" = "EDUC_9to11",
  "EDUC_4" = "EDUC_12_or_GED",
  "EDUC_5" = "EDUC_More_than_12",
  "MARSTAT_1" = "MARSTAT_Never_married",
  "MARSTAT_2" = "MARSTAT_Now_married",
  "MARSTAT_3" = "MARSTAT_Separated",
  "MARSTAT_4" = "MARSTAT_Divorced_widowed",
  "DEPRESSFLG_0" = "No_Depression",
  "DEPRESSFLG_1" = "Yes_Depression",
  "GENDER_1" = "GENDER_Male",
  "GENDER_2" = "GENDER_FEMALE"
)


colnames(dummy_vars) <- sapply(colnames(dummy_vars), function(col) {
  if (col %in% names(column_mapping)) {
    column_mapping[col]
  } else {
    col
  }
})

dummy_vars <- dummy_vars %>%
  select(-c(AGE_1, AGE_2, AGE_3, AGE_4, AGE_5, AGE_6, AGE_7, AGE_8, 
            AGE_9, AGE_10, AGE_11, AGE_12, AGE_13, AGE_14))

dummy_vars <- dummy_vars %>%
  select(-c(Political_Stance_NA))


dummy_vars <- dummy_vars %>%
  select(-matches("_-9"))

dummy_vars <- dummy_vars %>%
  select(-c(ETHNIC_1,ETHNIC_2,ETHNIC_3,ETHNIC_4))

# Logistic regression for Republican
dummy_vars$DEPRESSFLG <- as.factor(ifelse(dummy_vars$DEPRESSFLG == 1, 1, 0))

# Run logistic regression 
model <- glm(DEPRESSFLG ~ Political_Stance_Republican +  
               GENDER_FEMALE + GENDER_Male + MARSTAT_Never_married + MARSTAT_Now_married + 
               MARSTAT_Divorced_widowed + MARSTAT_Separated + EMPLOY_Part_time + 
               EMPLOY_Unemployed + EMPLOY_Full_time + EMPLOY_Not_in_labor_force + 
               EDUC_Special_education + EDUC_0to8 + EDUC_9to11 + EDUC_12_or_GED + 
               EDUC_More_than_12 + Children + Young_Adult + Adult + Middle_Aged_Adults + Senior + Hispanic + Non_Hispanic,
             data = dummy_vars, 
             family = binomial)

# Summary of the model
summary(model)

# Exponentiate the coefficient for Political_Stance_Republican to interpret odds ratio
exp(coef(model)["Political_Stance_Republican"])

#---------------------#

# Logistic regression for Democratic 
dummy_vars$DEPRESSFLG <- as.factor(ifelse(dummy_vars$DEPRESSFLG == 1, 1, 0))

model <- glm(DEPRESSFLG ~ Political_Stance_Democratic +  
               GENDER_FEMALE + GENDER_Male + MARSTAT_Never_married + MARSTAT_Now_married + 
               MARSTAT_Divorced_widowed + MARSTAT_Separated + EMPLOY_Part_time + 
               EMPLOY_Unemployed + EMPLOY_Full_time + EMPLOY_Not_in_labor_force + 
               EDUC_Special_education + EDUC_0to8 + EDUC_9to11 + EDUC_12_or_GED + 
               EDUC_More_than_12 + Children + Young_Adult + Adult + Middle_Aged_Adults + Senior + Hispanic + Non_Hispanic,
             data = dummy_vars, 
             family = binomial)

# Summary of the model
summary(model)

# Exponentiate the coefficient for Political_Stance_Republican to interpret odds ratio
exp(coef(model)["Political_Stance_Democratic"])



#---------------------------------------#

pred <- predict(model, dummy_vars, type = "response")
roc_curve <- roc(dummy_vars$DEPRESSFLG, pred)
plot(roc_curve)
auc(roc_curve)

# Ensure DEPRESSFLG is numeric
republican_group$DEPRESSFLG <- as.numeric(as.character(republican_group$DEPRESSFLG))
democratic_group$DEPRESSFLG <- as.numeric(as.character(democratic_group$DEPRESSFLG))

# Perform the t-test
t_test_result <- t.test(republican_group$DEPRESSFLG, democratic_group$DEPRESSFLG, 
                        alternative = "two.sided", 
                        var.equal = TRUE) # Set var.equal=TRUE if you assume equal variances

# Print the t-test results
print(t_test_result)