##### PART 1 EXPLORATORY DATA ANALYSIS (EDA)

### introduction to data set
data <- read.csv("heart-disease-dsa1101.csv")
head(data)
names(data)
dim(data)
attach(data)
set.seed(1101)

### load libraries
library(class)          # for KNN
library("rpart")        # for decision trees
library("rpart.plot")   # to plot decision trees
library(ROCR)           # for ROC curve

### association between heart disease status and age (categorical and quantitative)
data$disease = as.factor(data$disease)
boxplot(age ~ disease, col = "blue")
# association 
# 1. half of the boxplot area overlapping each other
# 2. the median of age with disease status = absence is slightly higher than the median with disease status = presence
# 3. the the IQR with disease = presence is larger than the IQR with disease = absence
# 4. there is possible outlier for the disease status = absence
# thus, there might be a quite strong association between heart disease status and age

### association between heart disease status and sex (categorical and categorical)
data$sex = as.factor(data$sex)
table = table(sex, disease) ; table
prop.table(table)                             
tab = prop.table(table, "disease") ; tab 
OR = (tab[1]/(1-tab[1]))/(tab[2]/(1-tab[2])); OR      # 0.04432133
# association
# 1. the odds of disease = absence among female is 0.0443 times the odds of disease = absence among male.
# 2. the odds of ratio is very low. hence, there is possible strong association between heart disease status and sex

### association between heart disease status and chest.pain (categorical and categorical)
data$chest.pain = as.factor(data$chest.pain)
count = table(chest.pain) ; count
barplot(count)
disease_prob = prop.table(table(disease)) ; disease_prob
chest.pain_prob = prop.table(table(disease, chest.pain), margin = 1) ; chest.pain_prob
# by conditional probability = 
# P (disease = 1 | chest.pain = 0) = 0.23456790 (typical angina)
# P (disease = 1 | chest.pain = 1) = 0.25308642 (atypical angina)
# P (disease = 1 | chest.pain = 2) = 0.41975309 (non-anginal pain)
# P (disease = 1 | chest.pain = 3) = 0.09259259 (asymptomatic)
# association 
# 1. the range of the probability of disease = 1 with some levels of chest pain is quite vary 
#    the probability of disease = 1 when chest.pain = 3 is only 0.09259259, while when chest.pain = 2 is 0.41975309 (around 4 times higher)
# 2. this indicates, there is possible association between heart disease status and chest-pain

### association between heart disease status and bp (categorical and quantitative)
boxplot(bp ~ disease, col = "blue")
# association
# 1. the median of bp for both disease = absence and disease = presence is similar around 130
# 2. the IQR for disease = absence is slightly larger than the IQR for disease = presence
# 3. there are some possible outliers for both disease = absence and disease = presence
# thus, there might be association between heart disease status and bp, but not too significant

### association between heart disease status and chol (categorical and quantitative)
boxplot(chol ~ disease, col = "blue")
# association
# 1. the median of chol for both disease = absence and disease = presence is similar around 250
# 2. the IQR for disease = absence is slightly larger than the IQR for disease = presence
# 3. the disease = presence shows more outliers than the disease = absence
# thus, there might be association between heart disease status and bp, but not too significant

### association between heart disease status and fbs (categorical and categorical)
data$fbs = as.factor(data$fbs)
table = table(fbs, disease) ; table
prop.table(table)                             
tab = prop.table(table, "disease") ; tab 
OR = (tab[1]/(1-tab[1]))/(tab[2]/(1-tab[2])); OR      # 27.80165
# association
# 1. the odds of disease = absence among fbs<120mg/dl is 27.802 times the odds of disease = absence among fbs>120mg/dl
# 2. the odds of ratio is very high. hence, there is possible strong association between heart disease status and fbs

### association between heart disease status and rest.ecg (categorical and categorical)
data$rest.ecg = as.factor(data$rest.ecg)
count = table(rest.ecg) ; count
barplot(count)
disease_prob = prop.table(table(disease)) ; disease_prob
rest.ecg_prob = prop.table(table(disease, rest.ecg), margin = 1) ; rest.ecg_prob
# by conditional probability = 
# P (disease = 1 | rest.ecg = 0) = 0.40740741 (normal)
# P (disease = 1 | rest.ecg = 1) = 0.58641975 (ST-T wave abnormality)
# P (disease = 1 | rest.ecg = 2) = 0.00617284 (possible or definite left ventricular hypertrophy)
# association 
# 1. the range of the probability of disease = 1 with some levels of rest.ecg is very vary 
#    the probability of disease = 1 when rest.ecg = 2 is only 0.00617284, while when rest.ecg = 1 is 0.58641975 (around 95 times higher)
# 2. this indicates, there is possible strong association between heart disease status and rest.ecg

### association between heart disease status and heart.rate (categorical and quantitative)
boxplot(heart.rate ~ disease, col = "blue")
# association
# 1. the disease = presence has higher median of heart rate than the disease = absence
# 2. the disease = presence shows more outliers than the disease = absence
# thus, there might be a strong association between heart disease status and heart rate

### association between heart disease status and angina (categorical and categorical)
data$angina = as.factor(data$angina)
table = table(angina, disease) ; table
prop.table(table)                             
tab = prop.table(table, "disease") ; tab 
OR = (tab[1]/(1-tab[1]))/(tab[2]/(1-tab[2])); OR      # 0.6655125
# association
# 1. the odds of disease = absence among angina = yes is 0.666 times the odds of disease = absence among angina = no
# 2. the odds of ratio is quite low. hence, there is possible association between heart disease status and angina

### association between heart disease status and st.depression (categorical and quantitative)
boxplot(st.depression ~ disease, col = "blue")
# association
# 1. the disease = presence has very low median of st.depression compared to disease = absence
# 2. the IQR of the disease = absence is approximately twice of the IQR of disease = presence
# 3. there are some possible outliers for the both of disease = presence and disease = absence
# thus, there might be a strong association between heart disease status and st.depression

### association between heart disease status and vessels (categorical and categorical)
data$vessels = as.factor(data$vessels)
count = table(vessels) ; count
barplot(count)
disease_prob = prop.table(table(disease)) ; disease_prob
vessels_prob = prop.table(table(disease, vessels), margin = 1) ; vessels_prob
# by conditional probability = 
# P (disease = 1 | vessels = 0) = 0.790123457 
# P (disease = 1 | vessels = 1) = 0.123456790 
# P (disease = 1 | vessels = 2) = 0.043209877 
# P (disease = 1 | vessels = 3) = 0.018518519 
# P (disease = 1 | vessels = 4) = 0.024691358 
# association 
# 1. the range of the probability of disease = 1 with some levels of vessels is very vary 
#    the probability of disease = 1 when vessels = 3 is only 0.018518519, while when vessels = 0 is 0.790123457 (around 40 times higher)
# 2. this indicates, there is possible strong association between heart disease status and vessels

### association between heart disease status and blood.disorder (categorical and categorical)
data$blood.disorder = as.factor(data$blood.disorder)
index = which(blood.disorder == 0)
data$blood.disorder[index] = 2
count = table(data$blood.disorder) ; count
barplot(count)
disease_prob = prop.table(table(disease)) ; disease_prob
blood.disorder_prob = prop.table(table(disease, data$blood.disorder), margin = 1) ; blood.disorder_prob
# by conditional probability = 
# P (disease = 1 | blood.disorder = 1) = 0.03703704 (normal)
# P (disease = 1 | blood.disorder = 2) = 0.79012346 (fixed defect)
# P (disease = 1 | blood.disorder = 3) = 0.17283951 (reversible defect)
# association 
# 1. the range of the probability of disease = 1 with some levels of blood.disorder is vary 
#    the probability of disease = 1 when blood.disorder = 1 is only 0.03703704, while when blood.disorder = 2 is 0.79012346 (around 20 times higher)
# 2. this indicates, there is possible association between heart disease status and blood.disorder


##### PART 2 METHODS: BUILDING MODEL/CLASSIFIER

### KNN
data$vessels = as.numeric(data$vessels)                 # change into numerical because the number indicates the level
data$blood.disorder = as.numeric(data$blood.disorder)   # change into numerical because the number indicates the level
X = data[, c(1, 4, 5, 8, 10, 11, 12)]
standardized.X = scale(X)

n_folds = 5
fold = sample(rep(1:5, length.out = dim(data)[1] )) 
sqrt(dim(data)[1])    # 17.32051 (so, the limit for k will be capped at 17)

k = 17
k_tpr = numeric(k)
for (i in seq(1, k, by = 2)){
  tpr = numeric(n_folds) 
  for (j in 1:n_folds) {
    test_index = which(fold == j)
    train.X = standardized.X[-test_index, ]
    test.X = standardized.X[test_index, ]
    train.Y = data$disease[-test_index] 
    test.Y = data$disease[test_index]
    pred = knn(train.X, test.X, train.Y, k = i) 
    confusion_matrix = table(pred, test.Y)
    tpr[j] = confusion_matrix[2,2] / (confusion_matrix[1,2] + confusion_matrix[2,2])
  }
  k_tpr[i] = mean(tpr) 
}

best_k = which(k_tpr == max(k_tpr)); best_k    
# 13 15 (so, the max tpr is when k = 13 and 15, the smaller k (k = 13) will be chosen)
max(k_tpr)   

k = 13
precision_fold = numeric(n_folds)
for (j in 1:n_folds) {
  test_index = which(fold == j)
  train.X = standardized.X[test_index, ]
  test.X = standardized.X[test_index, ]
  train.Y = data$disease[test_index] 
  test.Y = data$disease[test_index]
  pred = knn(train.X, test.X, train.Y, k = k)
  confusion_matrix = table(pred, test.Y)
  precision_fold[j] = confusion_matrix[2,2] / (confusion_matrix[2,1] + confusion_matrix[2,2])
}
precision_best_k = mean(precision_fold) ; precision_best_k         

pred = knn(standardized.X, standardized.X, data$disease, k = 13)
confusion_matrix = table(pred, data$disease)
tpr = confusion_matrix[2,2] / (confusion_matrix[1,2] + confusion_matrix[2,2]) ; tpr   
# TPR for KNN = 0.8641975
precision = confusion_matrix[2,2] / (confusion_matrix[2,1] + confusion_matrix[2,2]) ; precision  
# precision for KNN = 0.7954545

### DECISION TREES
minsplit = 1:30
length(minsplit)
dt_tpr = rep(0, length(minsplit))

for (i in 1:length(minsplit)) {
  tpr = numeric(0)
  for (j in 1:n_folds) {
    test <- which(fold == j)
    fit <- rpart(disease ~ age + sex + chest.pain + bp + chol + fbs + rest.ecg + heart.rate + angina + st.depression + vessels + blood.disorder,
              method="class", data=data, control=rpart.control(minsplit = i),   
              parms=list(split='information'))
    pred = predict(fit, data = data, type = "class")
    confusion_matrix = table(pred, data$disease)
    tpr = append(tpr, confusion_matrix[2,2] / (confusion_matrix[1,2] + confusion_matrix[2,2]))
  }
  dt_tpr[i] = mean(tpr) 
}

best_minsplit = minsplit[which(dt_tpr == max(dt_tpr))] ; best_minsplit
# 1  2  3  4  5  6  7  8  9  10(so, when minsplit 1 to 10, it gives the highest TPR)
max(dt_tpr)      

minsplit = best_minsplit
precision_fold = numeric(n_folds)
for (j in 1:n_folds) {
  test <- which(fold == j)
  fit <- rpart(disease ~ age + sex + chest.pain + bp + chol + fbs + rest.ecg + heart.rate + angina + st.depression + vessels + blood.disorder,
             method="class", data=data, control=rpart.control(minsplit = 10),   
             parms=list(split='information'))
  pred = predict(fit, data = data, type = "class")
  confusion_matrix = table(pred, data$disease)
  precision_fold[j] = confusion_matrix[2,2] / (confusion_matrix[2,1] + confusion_matrix[2,2])
}
precision_minsplit_10 = mean(precision_fold) ; precision_minsplit_10          

fit <- rpart(disease ~ age + sex + chest.pain + bp + chol + fbs + rest.ecg + heart.rate + angina + st.depression + vessels + blood.disorder,
             method="class", data=data, control=rpart.control(minsplit = 10),   
             parms=list(split='information'))
pred = predict(fit, data = data, type = "class")
confusion_matrix = table(pred, data$disease)
tpr_dt = confusion_matrix[2,2] / (confusion_matrix[1,2] + confusion_matrix[2,2]) ; tpr_dt
# TPR for decision tree = 0.9444444
precision_dt = confusion_matrix[2,2] / (confusion_matrix[2,1] + confusion_matrix[2,2]) ; precision_dt
# precision for decision tree = 0.8742857

rpart.plot(fit, type=4, extra=2, varlen=0, faclen=0, clip.right.labs=FALSE)


### LOGISTIC REGRESSION
M3 <- glm(disease ~., data = data,family = binomial)
summary(M3)

tab = table(data$disease)
prop.table(tab)
# proportion of disease = presence is 0.54 (54%), so we set the threshold = 0.54
# take the proportion of the success probability (disease = presence) as the borderline/threshold

prob.lr = predict(M3, type ="response")
delta = prop.table(tab)[2] 
data$disease_class_lr = ifelse (prob.lr >= delta, 1, 0)
confusion_matrix = table(actual = data$disease, predicted = data$disease_class_lr)

tpr_lr = confusion_matrix[2,2] / (confusion_matrix [2,2] + confusion_matrix[2,1]) ; tpr_lr               
# TPR for logistic regression = 0.8950617
precision_lr = confusion_matrix[2,2] / (confusion_matrix [2,2] + confusion_matrix[1,2]) ; precision_lr   
# precision for logistic regression = 0.8333333


##### PLOTTING ROC CURVE AND AUC VALUE

### ROC curve and AUC value for KNN
pred.knn = knn(standardized.X, standardized.X, data$disease, k = 13, prob = TRUE)
winning.prob = attr(pred.knn, "prob") 
n = length(data$disease)
prob.knn = numeric(n) 
for (i in 1:n) { prob.knn[i] = ifelse(pred.knn[i] == "1", winning.prob[i], 1-winning.prob[i]) } 
pred_knn = prediction(prob.knn, data$disease) 
roc_knn = performance(pred_knn , "tpr", "fpr")
auc_knn = performance(pred_knn , measure ="auc")
auc_knn@y.values[[1]]
# AUC value = 0.8985507
plot(roc_knn , col = "darkgreen", main = paste(" Area under the curve :", round(auc_knn@y.values[[1]] ,4)))

## ROC curve and AUC value for Decision Tree
predicted_probs_dt = predict(fit,newdata=data,type="prob")[,2]
actual_class <- data$disease == '1'
pred.dt <- prediction(predicted_probs_dt, actual_class)
perf.dt <- performance(pred.dt , "tpr", "fpr")
plot (perf.dt, lwd =2, col = "red", add = TRUE)
auc.dt <- performance(pred.dt , "auc")@y.values[[1]]
auc.dt    # 0.9228395

## ROC curve and AUC value for Logistic Regression
pred.lr = prediction(prob.lr , disease) 
roc.lr = performance(pred.lr , "tpr", "fpr")
auc.lr = performance(pred.lr , measure ="auc")
auc.lr@y.values[[1]]
# AUC value = 0.9248524
plot(roc.lr , col = "blue", main = paste(" Area under the curve :", round(auc.lr@y.values[[1]] ,4)), add = TRUE)

abline (a=0, b=1, col ="black", lty =3)
legend(0.5,0.4,legend=c("KNN", "Decision Tree", "Logistic Regression"),col=c("darkgreen", "red", "blue"), pch=c(20,20, 20))

### comments on pros and cons for each classifier fitted

## KNN
# Pros:
# 1. the distribution of the data is not considered, so it is suitable for wide range of data,
#    it is suitable for data with non-linear relationship as well
# 2. can have some choices of K so can choose the best K that gives the highest True Positive Rate (TPR)
# 3. simple algorithm, because KNN mainly predict the data based on similarity between nearest data points

# Cons:
# 1. the input variable strictly only for numerical variable, 
#    so the categorical variable cannot be considered even if it might has a significant effect to the prediction
# 2. need to use slightly more complex way to find the best K to fit the model
# 3. the precision value produced by KNN is the smallest compared to the other model (for this case)


## Decision Tree
# Pros:
# 1. model can be visualize as a tree, making it a very clear model and easy to interpret
# 2. can handle various data types (both categorical and numerical), so can have wider range of the input variables
# 3. no need assumptions for the data distribution, so can be used to perform a model with complex association
# 4. can have some choices of minsplit so can choose the best minsplit that gives the highest True Positive Rate (TPR)
# 5. the True Positive Rate and precision value produced by decision tree are the highest among the other model (for this case)

# Cons:
# 1. possibility to have an overfit model, especially when the value of minsplit is too low
# 2. the structure of the tree might be unstable when we make a small change in the data because will lead to different splits


## Logistic Regression
# Pros:
# 1. the model can provide coefficients for each variables and factors, making it easy to identify which variable is not significant
# 2. can handle various data types (both categorical and numerical), so can have wider range of the input variables
# 3. the AUC value produced by logistic regression is the highest compared to the other model (for this case)

# Cons:
# 1. there is an assumption of the data distribution (linearity), 
#    so, it is slightly not suitable for model with complex data and non-linear relationships between each variables
# 2. the response variable is binary, so we only can know whether the response variable is success or not


### best model/classifier
# summary of each model/classifier
#                           TPR           precision       AUC value
# KNN                    0.8641975        0.7954545       0.8985507
# Decision Tree          0.9444444        0.8742857       0.9228395
# Logistic Regression    0.8950617        0.8333333       0.9248524


# I will propose the Decision Tree as the best classifier to be used in predicting the heart disease status for this data set
# some reasons of why I choose Decision Tree:
# 1. the TPR and precision value produced by decision tree is far higher than the other two models (KNN and logistic regression)
#    it means the decision tree model predict the response variable more well compared to the other two models
#    the performance of the model is also very good 
# 2. while the AUC value of decision tree is lower than the AUC value of logistic regression, 
#    the difference is not too significant (only around 0.002 or 0.2%)
