
#####
# 
#   Financial Time Series Forecasting:
#   A Random Forest Approach
# 
#   Author: Ella Chen
#   



"""

  Goal of exercise: prediction market movement (+/-) wrt to previous day.

   ------------
    Input Data
   ------------

      Daily [Open, Close, High, Low, Volume] of S&P500 index (^GSPC) 
      from 2010/01/01 to 2015/11/24. 1485 data points after preprocessing.

      Source: Yahoo Finance.

   ------------
    Interest   
   ------------

      * Technical indicators as features
      * Feature selection
      * Hyperparameter tuning
      * Model evaluation

   ------------
    To-Do's  
   ------------

      * Handling highly correlated features
      * Clean up the code

"""

# seed
library(MASS)


# model
library(AUC)
library(tree)
library(caret)
library(maptree)
library(randomForest)
library(FSelector)


# Technical Trading Rules
library(TTR)
library(quantmod)   


# Plotting packages
library(ROCR)
library(ggplot2) 
library(graphics)      # Accomodate multiple plots
library(corrplot)      # Visualize corr between features
library(plotrix)       # Visualize RF prediction power and weights
library(timeSeries)    # time series visualization ts.plot


# --------------
#  Import data
# --------------

sp500 <- na.omit(getYahooData("^GSPC", 20100101, 20151124))
dim(sp500) #1485 x 5

# An overview of the data
chartSeries(sp500, theme="white")
addTA(OpCl(sp500),col='blue', type='h') 


# --------------
# Create labels
# --------------

# vector of all close prices
close_p <- as.vector(sp500$Close) 
length(close_p) # 1485

# daily change in close price
# a list of difference in close price
price_change <- c()
for (i in 1:1484){
  price_change <- append(price_change, close_p[i+1]-close_p[i])
}

# The plot below shows that the majority of price change falls 
# between -20 to 20 while only very few fall below -50 or go above 50.
ts.plot(price_change, ylab="Amount of Change", col="blue", main="Day-to-Day Change in Close Price")
abline(h=10, col="red", lty=2, lwd=2)
abline(h=-10, col="red", lty=2,lwd=2)

# Two classes: UP, DOWN
# Create a vector of labels
daily_difference <- c()
for (i in 1:1484){
  if (close_p[i+1] - close_p[i] >= 0){daily_difference <- append(daily_difference, "UP")}
  else if(close_p[i+1] - close_p[i] < 0){daily_difference <- append(daily_difference, "DOWN")}}
    
# Plot the distribution of classes
diff.freq <- table(daily_difference)
diff.freq

daily_difference <- append(daily_difference, NA)
length(daily_difference) # 1485, no label for the last data point


# --------------------------------
# Features: TECHNICAL INDICATORS
# --------------------------------

# Technical indicators, open, close, high, low, volume 
# will be used as features, and are subject to selection.
# Below section constructs feature vectors.


# MA Family: n = 5, 10, 15, 20
# 12 Features

# MA(20)
sma20 <- SMA(close_p, n=20) # SMA(20): vector
wma20 <- WMA(close_p, n=20) # WMA(20): vector
ema20 <- EMA(close_p, n=20) # EMA(20): vector

# MA(5)
sma5 <- SMA(close_p, n=5) # SMA(5): vector
wma5 <- WMA(close_p, n=5) # WMA(5): vector
ema5 <- EMA(close_p, n=5) # EMA(5): vector

# MA(10)
sma10 <- SMA(close_p, n=10) # SMA(10): vector
wma10 <- WMA(close_p, n=10) # WMA(10): vector
ema10 <- EMA(close_p, n=10) # EMA(10): vector

# MA(15)
sma15 <- SMA(close_p, n=15) # SMA(15): vector
wma15 <- WMA(close_p, n=15) # WMA(15): vector
ema15 <- EMA(close_p, n=15) # EMA(15): vector


# Stochastic %K: 2 features

# Stochastic oscillator is a momentum indicator 
# that relates the location of each day's close relative 
# to the high/low range over the past n periods.

stochOSC <- stoch(sp500[,c("High","Low","Close")], nFastK=14, nFastD=3)
stochK <- stochOSC$fastK # %K
stochK <- as.vector(stochK[,1]) # vector
stochD <- stochOSC$fastD # %D
stochD <- as.vector(stochD[,1]) # vector


# Moving Average Convergence Divergence (MACD): 1 feature

MACD_data <- MACD(close_p,nFast=10, nSlow=5)
MACD_vec <- MACD_data[,1] # vector


# Commodity Channel Index (CCI): 4 features

# Identify starting and ending trends
# Buy (sell) if CCI rises above 100 (falls below -100) and 
# sell (buy) when it falls below 100 (rises above -100).

CCI5 <- CCI(sp500[,c("High","Low","Close")], n=5) # n = 5
CCI5 <- as.vector(CCI5[,1])
CCI10 <- CCI(sp500[,c("High","Low","Close")], n=10) # n = 10
CCI10 <- as.vector(CCI10[,1])
CCI15 <- CCI(sp500[,c("High","Low","Close")], n=15) # n = 15
CCI15 <- as.vector(CCI15[,1])
CCI20 <- CCI(sp500[,c("High","Low","Close")], n=20) # n = 20
CCI20 <- as.vector(CCI20[,1])


# DPO: 4 features
# Detrended Price Oscillator (DPO) removes the trend in prices

priceDPO5 <- DPO(sp500[,"Close"], n=5) # n=5
priceDPO5 <- as.vector(priceDPO5[,1])
priceDPO10 <- DPO(sp500[,"Close"], n=10) # n=10
priceDPO10 <- as.vector(priceDPO10[,1])
priceDPO15 <- DPO(sp500[,"Close"], n=15) # n=15
priceDPO15 <- as.vector(priceDPO15[,1])
priceDPO20 <- DPO(sp500[,"Close"], n=20) # n=20
priceDPO20 <- as.vector(priceDPO20[,1])


# Rate of change (ROC): 4 features

ROC5 <- ROC(sp500[,"Close"], n=5) # n = 5
ROC5 <- as.vector(ROC5[,1])
ROC10 <- ROC(sp500[,"Close"], n=10) # n = 10
ROC10 <- as.vector(ROC10[,1])
ROC15 <- ROC(sp500[,"Close"], n=15) # n = 15
ROC15 <- as.vector(ROC15[,1])
ROC20 <- ROC(sp500[,"Close"], n=20) # n = 20
ROC20 <- as.vector(ROC20[,1])


# momentum: 4 features

momentum5 <- momentum(sp500[,"Close"], n=5, na.pad=TRUE)
momentum5 <- as.vector(momentum5[,1])
momentum10 <- momentum(sp500[,"Close"], n=10, na.pad=TRUE)
momentum10 <- as.vector(momentum10[,1])
momentum15 <- momentum(sp500[,"Close"], n=15, na.pad=TRUE)
momentum15 <- as.vector(momentum15[,1])
momentum20 <- momentum(sp500[,"Close"], n=20, na.pad=TRUE)
momentum20 <- as.vector(momentum20[,1])


# Relative strength index (RSI): 4 features

# The Relative Strength Index (RSI) calculates a ratio of the recent 
# upward price movements to the absolute price movement. 

RSI5 <- RSI(sp500[,"Close"], n=5)
RSI5 <- as.vector(RSI5[,1])
RSI10 <- RSI(sp500[,"Close"], n=10)
RSI10 <- as.vector(RSI10[,1])
RSI15 <- RSI(sp500[,"Close"], n=15)
RSI15 <- as.vector(RSI15[,1])
RSI20 <- RSI(sp500[,"Close"], n=20)
RSI20 <- as.vector(RSI20[,1])


# 5-day close-to-close volatility

volatility5 <- volatility(close_p, n=5, calc="close")
volatility10 <- volatility(close_p, n=10, calc="close")
volatility15 <- volatility(close_p, n=15, calc="close")
volatility20 <- volatility(close_p, n=20, calc="close")


# Chande Momentum Oscillator CMO: 3 features

# Divides total movement by net movement
# Indicator of overbought and oversold conditions
# High CMo = strong trends
# Gives buy/sell signal when crosses above/below a moving average

CMO20 <- as.vector(CMO(sp500[,"Close"], n=20)) # n = 20
CMO5 <- as.vector(CMO(sp500[,"Close"], n=5)) # n = 5
CMO10 <- as.vector(CMO(sp500[,"Close"], n=10)) # n = 10
CMO15 <- as.vector(CMO(sp500[,"Close"], n=15)) # n = 15


# Welles Wilder's Directional Movement Index: 2 Features
# Direction movement index, a measure of trend and strength

ADX_ema <- ADX(sp500[,c("High","Low","Close")], maType=EMA) # EMA
ADX_ema <- as.vector(ADX_ema$ADX)
ADX_sma <- ADX(sp500[,c("High","Low","Close")], maType=SMA) # SMA
ADX_sma <- as.vector(ADX_sma$ADX)


# Aroon oscilator: 4 features

# Identify starting trends and measures how long it has been since 
# the highest high/lowest low has occurred in the last n periods

aroon_5 <- aroon(sp500[,c("High","Low")], n=5) # n=5
aroon5 <- as.vector(aroon_5$oscillator)
aroon_10 <- aroon(sp500[,c("High","Low")], n=10) # n=10
aroon10 <- as.vector(aroon_10$oscillator)
aroon_15 <- aroon(sp500[,c("High","Low")], n=15) # n=15
aroon15 <- as.vector(aroon_15$oscillator)
aroon_20 <- aroon(sp500[,c("High","Low")], n=20) # n=20
aroon20 <- as.vector(aroon_20$oscillator)


# Vertical Horizontal Filter: 4 features
# attempts to identify starting and ending trends

VHF5 <-as.vector(VHF(sp500[,c("High","Low","Close")], n=5)) # n=5
VHF10 <-as.vector(VHF(sp500[,c("High","Low","Close")], n=10)) # n=10
VHF15 <-as.vector(VHF(sp500[,c("High","Low","Close")], n=15)) # n=15
VHF20 <-as.vector(VHF(sp500[,c("High","Low","Close")], n=20)) # n=20


# Trend Detection Index: 4 features
# attempts to identify starting and ending trends

TDI5 <- TDI(sp500[,"Close"], n=5) # n = 5
TDI5 <- as.vector(TDI5$tdi)
TDI10 <- TDI(sp500[,"Close"], n=10) # n = 10
TDI10 <- as.vector(TDI10$tdi)
TDI15 <- TDI(sp500[,"Close"], n=15) # n = 15
TDI15 <- as.vector(TDI15$tdi)
TDI20 <- TDI(sp500[,"Close"], n=20) # n = 20
TDI20 <- as.vector(TDI20$tdi)


# Money Flow Index: 4 features
# ratio of positve and negative money flow over time

MFI5 <- as.vector(MFI(sp500[,c("High","Low","Close")], sp500[,"Volume"], n=5))
MFI10 <- as.vector(MFI(sp500[,c("High","Low","Close")], sp500[,"Volume"], n=10))
MFI15 <- as.vector(MFI(sp500[,c("High","Low","Close")], sp500[,"Volume"], n=15))
MFI20 <- as.vector(MFI(sp500[,c("High","Low","Close")], sp500[,"Volume"], n=20))


# Bollinger Bands on Close: 4 features
# Compare a security's volatility and price levels over a period of time

BBC5 <- BBands(sp500[,c("High","Low","Close")], n=5) # n=5
BBC5 <- as.vector(BBC5$mavg) 
BBC10 <- BBands(sp500[,c("High","Low","Close")], n=10) # n=10
BBC10 <- as.vector(BBC10$mavg) 
BBC15 <- BBands(sp500[,c("High","Low","Close")], n=15) # n=15
BBC15 <- as.vector(BBC15$mavg) 
BBC20 <- BBands(sp500[,c("High","Low","Close")], n=20) # n=20
BBC20 <- as.vector(BBC20$mavg) 


# ---------------------
# Standard features
# ---------------------

# High, Low, Open, Close, Volume

High <- as.vector(sp500$High)
Low <- as.vector(sp500$Low)
Open <- as.vector(sp500$Open)
Close <- as.vector(sp500$Close)
Vol <- as.vector(sp500$Volume)


# ---------------------
# Input Matrix
# ---------------------

# Consolidate all the features and label and remove N/A's.
# This set contains both the training data and testing data.

sp500.df <- na.omit(data.frame(High, Low, Open, Close, Vol, 
                               sma5, wma5, ema5, sma10, wma10, ema10,
                               sma15, wma15, ema15, sma20, wma20, ema20,
                               stochK, stochD,
                               MACD_vec,
                               CCI5, CCI10, CCI15, CCI20,
                               priceDPO5, priceDPO10, priceDPO15, priceDPO20,
                               ROC5, ROC10, ROC15, ROC20,
                               momentum5, momentum10, momentum15, momentum20,
                               RSI5, RSI10, RSI15, RSI20,
                               volatility5, volatility10, volatility15, volatility20,
                               CMO5, CMO10, CMO15, CMO20,
                               ADX_ema, ADX_sma,
                               aroon5, aroon10, aroon15, aroon20,
                               VHF5, VHF10, VHF15, VHF20,
                               TDI5, TDI10, TDI15, TDI20,
                               MFI5, MFI10, MFI15, MFI20,
                               BBC5, BBC10, BBC15, BBC20,
                               daily_difference))

dim(sp500.df) # 1435 x 71: label included as last column

# Initial training set: first 1000 points
training <- sp500.df[1:1000,]
dim(training) # 1000 x 71

testing <- sp500.df[1001:1435,]
dim(testing) # 435 x 71

sp500_nolabel <- sp500.df[,1:70] # no labels
dim(sp500_nolabel) # 1435 x 70


# ------------------------------
# Correlation between features
# ------------------------------

# COMBAK:
# Some of the features are likely highly correlated.
# Highly correlated variables cause bias in feature selection.


# Visualize the correlation using corrplot package

M <- cor(sp500_nolabel)
corrplot(M, method="color", tl.cex=0.5, tl.col="darkblue")


# ----------------------------------
# Preliminary model and evaluation
# ----------------------------------

# All features included
# No restriction on depth of trees

rf.sp500_1 <- randomForest(daily_difference ~ ., data=training)
rf.sp500_1 

# OOB 25.3%
# DOWN class error 0.2988764
# UP 0.2162162

rf.pred_1 <- predict(rf.sp500_1, testing, type="class")
summary(rf.pred_1)
table(rf.pred_1, testing$daily_difference)

TP_1 <- 165 # true positive
FP_1 <- 65 # false positve
TN_1 <- 167 # true negative
FN_1 <- 38 # false negative

Accuracy_1 <- (TP_1 + TN_1)/(TP_1 + TN_1 + FP_1 + FN_1)

Precision_p_1 <- TP_1/(TP_1+FP_1)
Recall_p_1 <- TP_1/(TP_1 + FN_1)

Precision_n_1 <- TN_1/(TN_1+FN_1)
Recall_n_1 <- TN_1/(TN_1 + FP_1)

# F score

F_positive_1 <- 2*((Precision_p_1*Recall_p_1)/(Precision_p_1+Recall_p_1))
F_negative_1 <- 2*((Precision_n_1*Recall_n_1)/(Precision_n_1+Recall_n_1))

pred_eval_1 <- matrix(c(Accuracy_1,F_positive_1,F_negative_1),ncol=3,byrow=TRUE)
pred_eval_1 <- data.frame(pred_eval_1)

names(pred_eval_1) <- c("Accuracy","F_positive","F_negative")
pred_eval_1

# The preliminary result show that there is a large area for improvement


# ---------------------
# Feature Selection
# ---------------------

# There are two problems with the data:
#   1. relatively high number of features (>70)
#   2. features are highly correlated

# Feature selection methods:
# 
#     1. Filter by FSelector 
#     2. rf.importance
#     3. Information gain


require(plyr)
setwd("~/projects/rf_exercise")
weights <- random.forest.importance(daily_difference~., training, importance.type=1)
weights <- data.frame(weights)

write.csv(weights, file="weights_unsorted.csv")
rf_filter <- read.csv(file="weights_sorted.csv")

# plot barplot for the ranking
barplot(rf_filter$attr_importance, names.arg=rf_filter$Feature, 
        horiz=TRUE, las=2, cex.names=0.5, border=NA, col="skyblue",
        main="FSelector Random Forest Feature Importance Ranking",
        xlab="Feature Importance",
        ylab="Feature")

# Use features with importance > 4 to train the model
rf_filter_48 <- rf_filter[1:48,]
rownames(rf_filter_48) <- rf_filter_48[,1]

rf_filter_48[,1] <- NULL
rf_filter_48_names <- row.names(rf_filter_48)

# Form new training data with 48 features: 1435 x 49, label included
IG.df <- na.omit(sp500.df[,c(rf_filter_48_names,"daily_difference")])
IG_training <- IG.df[1:1000,]
IG_testing <- IG.df[1001:1435,]

# Fit random forest
rf.IG <- randomForest(daily_difference ~ ., data=IG_training)
rf.IG 


rf.pred_IG <- predict(rf.IG, IG_testing, type="class")

summary(rf.pred_IG)
table(rf.pred_IG, IG_testing$daily_difference)


# -------------------------
# randomForest
# Mean Decrease Accuracy
# Mean Decrease Gini
# -------------------------

# Reference: Elements of Statistical Learning P593

# Mean Decrease Accuracy:
# Measures the impact of each feature on accuracy of the model
# Idea: permute the values of each feature and measure how much the permutation
# decreases the accuracy of the model. For unimportant variables, the permutation
# should have little to no effect on model accuracy, while permuting important features
# should significantly decrease it.

# Rank by variable importance and Gini importance
sp500.rf <- randomForest(daily_difference ~ ., data=training, ntree=1000, 
                          keep.forest=FALSE, importance=TRUE)

mean_decrease_accuracy <- importance(sp500.rf, type=1)
write.csv(mean_decrease_accuracy, file="mean_decrease_accuracy_rank.csv")
mean_decrease_accuracy_sorted <- read.csv(file="mean_decrease_accuracy_rank.csv")

mga_filter_48 <- mean_decrease_accuracy_sorted[1:48,]
rownames(mga_filter_48) <- mga_filter_48[,1]
mga_filter_48[,1] <- NULL
mga_filter_48_names <- row.names(mga_filter_48)

# Form new training data with 48 features for MGA
# 1435 x 49, label included

MGA.df <- na.omit(sp500.df[,c(mga_filter_48_names,"daily_difference")])
dim(MGA.df)

MGA_training <- MGA.df[1:1000,]
MGA_testing <- MGA.df[1001:1435,]

# Fit random forest
rf.MGA <- randomForest(daily_difference ~ ., data=MGA_training)
rf.MGA 


rf.pred_MGA <- predict(rf.MGA, MGA_testing, type="class")
summary(rf.pred_MGA)
table(rf.pred_MGA, MGA_testing$daily_difference)


# Mean Decrease Gini:

# Every time a split of a node is made on variable m the gini impurity criterion for the two 
# descendent nodes is less than the parent node. Adding up the gini decreases for each individual 
# variable over all trees in the forest gives a fast variable importance that is often very 
# consistent with the permutation importance measure.

mean_decrease_gini <- importance(sp500.rf, type=2)
write.csv(mean_decrease_gini, file="mean_decrease_gini_rank.csv")
mean_decrease_gini_sorted <- read.csv(file = "mean_decrease_gini_rank.csv")

mdg_filter_48 <- mean_decrease_gini_sorted[1:48,]
rownames(mdg_filter_48) <- mdg_filter_48[,1]
mdg_filter_48[,1] <- NULL
mdg_filter_48_names <- row.names(mdg_filter_48)

# Form new training data with 48 features for MDG
# 1435 x 49, label included

MDG.df <- na.omit(sp500.df[,c(mdg_filter_48_names,"daily_difference")]) 
dim(MDG.df)
MDG_training <- MDG.df[1:1000,]
MDG_testing <- MDG.df[1001:1435,]

# Fit random forest
rf.MDG <- randomForest(daily_difference ~ ., data=MDG_training)
rf.MDG 


rf.pred_MDG <- predict(rf.MDG, MDG_testing, type="class")
summary(rf.pred_MDG)
table(rf.pred_MDG, MDG_testing$daily_difference)


# Plot the importance graph
varImpPlot(sp500.rf, col="red", cex=0.65, main="Random Forest Feature Ranking", pch=15)


# -------------------------------
# Caret
# Recursive Feature Elimination
# -------------------------------
# 
# COMBAK:
#
# control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# # run the RFE algorithm
# rfe_ranking <- rfe(training[,1:70], training[,71], sizes=70, rfeControl=control)
# # summarize the results
# print(rfe_ranking)
# # list the chosen features
# predictors(rfe_ranking)
# # plot the results
# plot(rfe_ranking, type=c("g", "o"), main = "RFE Number of Features vs. Accuracy")
# rfe_overall_rank <- varImp(rfe_ranking$fit)
# write.csv(rfe_overall_rank, file = "RFE_overall_rank.csv")


# -----------------------
# Normalize rankings
# Write to CSV
# -----------------------

# Normalize the rankings for comparison accross different ranking methods


# Information Gain
info_gain_normalized <- scale(weights)
colMeans(scale(info_gain_normalized))
apply(scale(info_gain_normalized), 2, sd)
write.csv(info_gain_normalized, file="Information_Gain_Normalized.csv")

# Random Forest Rankings
mda_normalized <- scale(mean_decrease_accuracy)
colMeans(scale(mda_normalized))
apply(scale(mda_normalized), 2, sd)
write.csv(mda_normalized, file="Mean_Decrease_Accuracy_Normalized.csv")

mdg_normalized <- scale(mean_decrease_gini)
colMeans(scale(mdg_normalized))
apply(scale(mdg_normalized), 2, sd)
write.csv(mdg_normalized, file="Mean Decrease Gini Normalized.csv")

# rfe_normalized <- scale(rfe_overall_rank)
# colMeans(scale(rfe_normalized))
# apply(scale(rfe_normalized), 2, sd)
# write.csv(rfe_normalized, file = "RFE_overall_rank_normalized.csv")

# All above files are aggregated to compare the average
aggregate_feature_selection <- read.csv(file="aggregated_comparison_for_feature_selection.csv", header=TRUE)
aggregate_non_zero <- aggregate_feature_selection[1:34,]

rownames(aggregate_non_zero) <- aggregate_non_zero[,1]
aggregate_non_zero[,1] <- NULL
aggregate_non_zero_names <- row.names(aggregate_non_zero)
aggregate_non_zero_names

# Form new training data with 34 features for the aggregated ranking
# 1435 x 49, label included

aggregate.df <- na.omit(sp500.df[,c(aggregate_non_zero_names,"daily_difference")]) 
dim(aggregate.df)
aggregate_training <- aggregate.df[1:1000,]
aggregate_testing <- aggregate.df[1001:1435,]

# Fit random forest
rf.aggregate <- randomForest(daily_difference ~ ., data=aggregate_training)
rf.aggregate 


rf.pred_aggregate <- predict(rf.aggregate, aggregate_testing, type="class")
summary(rf.pred_aggregate)
table(rf.pred_aggregate, aggregate_testing$daily_difference)

# barplot(aggregate_non_zero$Mean, names.arg=aggregate_non_zero$Feature, 
#         horiz=TRUE, las=2, cex.names=0.5, border=NA, col = "orange",
#         main = "Aggregate Feature Selection",
#         xlab = "Averaged Feature Importance",
#         ylab = "Feature")


#-----------------------
# Wraper method: RFE
#-----------------------

# RFE on full set of features
control <- rfeControl(functions=rfFuncs, method="cv", number=10)

# # run the RFE algorithm
# rfe.full_features <- rfe(training[,1:70], training[,71], sizes=c(1:70), rfeControl=control)
# # summarize the results
# print(rfe.full_features)
# # list the chosen features
# predictors(rfe.full_features)
# # plot the results
# plot(rfe.full_features, type=c("g", "o"), main = "RFE Number of Features vs. Accuracy")
# rfe_overall_rank_reduced_full = varImp(rfe_ranking_reduced$fit)


# RFE on full reduced set of features

# run the RFE algorithm
rfe.reduced_features <- rfe(aggregate_training[,1:34], aggregate_training[,35], sizes=c(1:34), rfeControl=control)

# summarize the results
print(rfe.reduced_features)

# list the chosen features
predictors(rfe.reduced_features)

# plot the results
plot(rfe.reduced_features, type=c("g", "o"), main = "RFE Number of Features vs. Accuracy")

rfe_14_features <- predictors(rfe.reduced_features)

# Use the 14 features suggested by RFE based on the 34 features after the filter to build default model
rfe_14.df <- na.omit(sp500.df[,c(rfe_14_features,"daily_difference")]) # 1435 x 15, label included
dim(rfe_14.df)

rfe_14_training <- rfe_14.df[1:1000,]
rfe_14_testing <- rfe_14.df[1001:1435,]

# Fit random forest
rf.rfe_14 <- randomForest(daily_difference ~ ., data=rfe_14_training)
rf.rfe_14 


rf.pred_rfe_14 <- predict(rf.rfe_14, rfe_14_testing, type="class")
summary(rf.pred_rfe_14)
table(rf.pred_rfe_14, rfe_14_testing$daily_difference)

# # Make prediction on the first 100 instances in the testing set
# rf.pred_rfe_14_100 <- predict(rf.rfe_14, rfe_14_testing[1:100,], type="class")
# summary(rf.pred_rfe_14_100)
# table(rf.pred_rfe_14_100, rfe_14_testing[1:100,]$daily_difference)
# 
# # Make prediction on the second 100 instances in the testing set
# rf.pred_rfe_14_200 <- predict(rf.rfe_14, rfe_14_testing[101:200,], type="class")
# summary(rf.pred_rfe_14_200)
# table(rf.pred_rfe_14_200, rfe_14_testing[101:200,]$daily_difference)
# 
# # Make prediction on the third 100 instances in the testing set
# rf.pred_rfe_14_300 <- predict(rf.rfe_14, rfe_14_testing[201:300,], type="class")
# summary(rf.pred_rfe_14_300)
# table(rf.pred_rfe_14_300, rfe_14_testing[201:300,]$daily_difference)
# 
# # Make prediction on the fourth 100 instances in the testing set
# rf.pred_rfe_14_400 <- predict(rf.rfe_14, rfe_14_testing[301:400,], type="class")
# summary(rf.pred_rfe_14_400)
# table(rf.pred_rfe_14_400, rfe_14_testing[301:400,]$daily_difference)

# From the results we can see that the farther away the test data is from the training set, the harder it is
# to make accurate prediction. We may consider using sliding window on the training data.

# Come back to this after model tuning.
# By intuition it is better to add new data to the training set instead of using complete sliding window
# since business cycles can repeat and same patterns in the past can show up again.

# Predictors we have so far:
# Preliminary predictor (70 features): rf.pred_1

# par(mfrow=c(1,1))
# plot(roc(rf.sp500_1$votes[,1]), rf.sp500_1$y == "UP",main="ROC curves for four models predicting class 0")
# plot(roc(rf2$votes[,1],factor(1 * (rf1$y==0))),col=2,add=T)
# plot(roc(rf3$votes[,1],factor(1 * (rf1$y==0))),col=3,add=T)
# plot(roc(rf4$votes[,1],factor(1 * (rf1$y==0))),col=4,add=T)


# rf1 <- randomForest (x,y,...);
# OOB.votes <- predict(rf.sp500_1,testing,type="prob");
# OOB.pred <- OOB.votes[,2];
# 
# pred.obj <- prediction (OOB.pred, testing$daily_difference);
# 
# RP.perf <- performance(pred.obj, "rec","prec");
# plot (RP.perf);
# 
# ROC.perf <- performance(pred.obj, "fpr","tpr");
# plot (ROC.perf);
# 
# plot  (RP.perf@alpha.values[[1]],RP.perf@testing.values[[1]]);
# lines (RP.perf@alpha.values[[1]],RP.perf@y.values[[1]]);
# lines (ROC.perf@alpha.values[[1]],ROC.perf@x.values[[1]]);

#-----------------
# Model tuning
#-----------------

rfe_14_features

# From the above feature selection process we have chosen 14 features
# "priceDPO5"  "priceDPO10" "priceDPO15" "priceDPO20" "RSI5"      
# "CCI15"      "CMO10"      "CCI5"       "ROC10"      "momentum10"
# "stochK"     "MFI15"      "RSI10"      "CCI10"  

# Further tune this model to find the best mtry, number of samples, trees,
# and proportion of data points to sample from each classes. 
# Then see recursive variable importance pruning may lead to higher
# prediction power.
# 
# # Use the 14 features suggested by RFE based on the 34 features after the filter to build default model
# rfe_14.df <- na.omit(sp500.df[,c(rfe_14_features,"daily_difference")]) # 1435 x 15, label included
# dim(rfe_14.df)
# rfe_14_training <- rfe_14.df[1:1000,]
# rfe_14_testing <- rfe_14.df[1001:1435,]
# 
# # Fit random forest
# rf.rfe_14 <- randomForest(daily_difference ~ ., data=rfe_14_training)


# Tune the default model: 
# mtry = 1:14, number of trees 500, 800, 1000

OOB_error <- c()
UP_error <- c()
DOWN_error <- c()
T_UP <- c()
T_DOWN <- c()
F_UP <- c()
F_DOWN <- c()

for (mtry in 1:14) {

  for (ntree in c(500, 800, 1000)){
    rf.fit_no_sub <- randomForest(daily_difference ~ ., data=rfe_14_training, 
                                  mtry=mtry, ntree = ntree)

    plot(rf.fit_no_sub, main=paste("Number of trees v.s. Error at mtry = ", mtry))
    
    rf.pred_no_sub <- predict(rf.fit_no_sub, rfe_14_testing, type="class")

    print(paste("mtry = ", mtry))
    print(paste("ntree = ", ntree))
    print(rf.fit_no_sub$err.rate[ntree,])
    print(table(rf.pred_no_sub, rfe_14_testing$daily_difference))

    OOB_error <- append(OOB_error, rf.fit_no_sub$err.rate[ntree,1])
    UP_error <- append(UP_error, rf.fit_no_sub$err.rate[ntree,3])
    DOWN_error <- append(DOWN_error, rf.fit_no_sub$err.rate[ntree,2])
    T_UP <- append(T_UP, table(rf.pred_no_sub, rfe_14_testing$daily_difference)[2,2])
    F_UP <- append(F_UP, table(rf.pred_no_sub, rfe_14_testing$daily_difference)[1,2])
    T_DOWN <- append(T_DOWN, table(rf.pred_no_sub, rfe_14_testing$daily_difference)[1,1])
    F_DOWN <- append(F_DOWN, table(rf.pred_no_sub, rfe_14_testing$daily_difference)[2,1])

  }
}

# Data frame of the above tuning outputs

tuning_df <- data.frame(data.frame(OOB_error),data.frame(UP_error),data.frame(DOWN_error), 
                        data.frame(T_UP), data.frame(F_UP), data.frame(T_DOWN), data.frame(F_DOWN))

write.csv(tuning_df, file="tuning results.csv")

# The output shows that mtry = 5 and ntree = 500 can slightly improve the prediction accuracy

# Imbalanced data issue

# Tune the default 14-variable model to find a good sample size from each class
# Use mtry = 5 and ntree = 500

# Investigate if the imbalanced data and different choice of sampling proportions 
# impose an impact on the votings


# No stratification
rf1 <- randomForest(daily_difference~.,data=rfe_14_training, mtry=5, ntree=500, sampsize=500)

# UP 1: DOWN 1.5
rf2 <- randomForest(daily_difference~.,data=rfe_14_training, mtry=5, ntree=500,
                    sampsize=c(200, 300),strata=rfe_14_training$daily_difference)

# UP 1.5: DOWN 1
rf3 <- randomForest(daily_difference~.,data=rfe_14_training, mtry=5, ntree=500,
                    sampsize=c(300, 200),strata=rfe_14_training$daily_difference)
# UP 1: DOWN 1
rf4 <- randomForest(daily_difference~.,data=rfe_14_training, mtry=5, ntree=500,
                    sampsize=c(250, 250),strata=rfe_14_training$daily_difference)

# COMBAK:
# plot.separation <- function(rf,...) {
#   triax.plot(rf$votes,...,col.symbols = c("#FF0000FF",
#                                           "#00FF0010",
#                                           "#0000FF10")[as.numeric(rf$y)])
# }

# plot.separation(rf1,main="no stratification")
# plot.separation(rf2,main="UP 1: DOWN 1.5")
# plot.separation(rf3,main="UP 1.5: DOWN 1")
# plot.separation(rf4,main="UP 1: DOWN 1")
# rfnew <- randomForest(daily_difference ~ ., data=training, sampsize=c(500, 500), strata=training$daily_difference)
# plot(roc(rfnew$votes[,1], factor(1*(rf.sp5000$daily_difference == 0))), col = "black", add = T)

# Notice that the rf4 has the most balanced error rate for the two classes. 
# Tune by sampling the same amount of points from both classes.
# Identify the number of observations to sample from each class:

OOB_s <- c()
UP_error_s <- c()
DOWN_error_s <- c()
T_UP_s <- c()
T_DOWN_s <- c()
F_UP_s <- c()
F_DOWN_s <- c()

for (i in c(50, 100, 150, 200, 250, 300, 350, 400)){
  
  rf_stratified <- randomForest(daily_difference~.,data=rfe_14_training, mtry=5, ntree=500,
                                sampsize=c(i, i),strata=rfe_14_training$daily_difference)

  rf.pred_stratified <- predict(rf_stratified, rfe_14_testing, type="class")
  
  print(paste("# samples for each class = ", i))

  OOB_s <- append(OOB_s, rf_stratified$err.rate[500,1])
  UP_error_s <- append(UP_error_s, rf_stratified$err.rate[500,3])
  DOWN_error_s <- append(DOWN_error_s, rf_stratified$err.rate[500,2])
  T_UP_s <- append(T_UP_s, table(rf.pred_stratified, rfe_14_testing$daily_difference)[2,2])
  F_UP_s <- append(F_UP_s, table(rf.pred_stratified, rfe_14_testing$daily_difference)[1,2])
  T_DOWN_s <- append(T_DOWN_s, table(rf.pred_stratified, rfe_14_testing$daily_difference)[1,1])
  F_DOWN_s <- append(F_DOWN_s, table(rf.pred_stratified, rfe_14_testing$daily_difference)[2,1])

}

s_tuning_df <- data.frame(data.frame(OOB_s),data.frame(UP_error_s),data.frame(DOWN_error_s), 
                          data.frame(T_UP_s), data.frame(F_UP_s), data.frame(T_DOWN_s), data.frame(F_DOWN_s))

write.csv(s_tuning_df, file="stratified_tuning_results.csv")

# To compare the model performances, plot ROC curves
# 50, 100, 150, 250, 350, 400

rf50 <- randomForest(daily_difference~.,data=rfe_14_training, mtry=5, ntree=500,
                     sampsize=c(50, 50),strata= rfe_14_training$daily_difference)

rf100 <- randomForest(daily_difference~.,data=rfe_14_training, mtry=5, ntree=500,
                     sampsize=c(100, 100),strata=rfe_14_training$daily_difference)

rf150 <- randomForest(daily_difference~.,data=rfe_14_training, mtry=5, ntree=500,
                     sampsize=c(150, 150),strata=rfe_14_training$daily_difference)

rf250 <- randomForest(daily_difference~.,data=rfe_14_training, mtry=5, ntree=500,
                     sampsize=c(250, 250),strata=rfe_14_training$daily_difference)

rf350 <- randomForest(daily_difference~.,data=rfe_14_training, mtry=5, ntree=500,
                     sampsize=c(350, 350),strata=rfe_14_training$daily_difference)

rf400 <- randomForest(daily_difference~.,data=rfe_14_training, mtry=5, ntree=500,
                     sampsize=c(400, 400),strata=rfe_14_training$daily_difference)


rf_50_test <- predict(rf50, type="prob", newdata=rfe_14_testing)
forestpred_50 <- prediction(rf_50_test[,2], rfe_14_testing$daily_difference)
perf_50 <- performance(forestpred_50, "tpr", "fpr")

rf_100_test <- predict(rf100, type="prob", newdata=rfe_14_testing)
forestpred_100 <- prediction(rf_100_test[,2], rfe_14_testing$daily_difference)
perf_100 <- performance(forestpred_100, "tpr", "fpr")

rf_150_test <- predict(rf150, type="prob", newdata=rfe_14_testing)
forestpred_150 <- prediction(rf_150_test[,2], rfe_14_testing$daily_difference)
perf_150 <- performance(forestpred_150, "tpr", "fpr")

rf_250_test <- predict(rf250, type="prob", newdata=rfe_14_testing)
forestpred_250 <- prediction(rf_250_test[,2], rfe_14_testing$daily_difference)
perf_250 <- performance(forestpred_250, "tpr", "fpr")

rf_350_test <- predict(rf350, type="prob", newdata=rfe_14_testing)
forestpred_350 <- prediction(rf_350_test[,2], rfe_14_testing$daily_difference)
perf_350 <- performance(forestpred_350, "tpr", "fpr")

rf_400_test <- predict(rf400, type="prob", newdata=rfe_14_testing)
forestpred_400 <- prediction(rf_400_test[,2], rfe_14_testing$daily_difference)
perf_400 <- performance(forestpred_400, "tpr", "fpr")

rf_unstratified_test <- predict(rf1, type="prob", newdata=rfe_14_testing)
forestpred_unstratified <- prediction(rf_unstratified_test[,2], rfe_14_testing$daily_difference)
perf_unstratified <- performance(forestpred_unstratified, "tpr", "fpr")

rf_fullset_test <- predict(rf.sp500_1, type="prob", newdata=testing)
forestpred_fullset <- prediction(rf_fullset_test[,2], testing$daily_difference)
perf_fullset <- performance(forestpred_fullset, "tpr", "fpr")


#plot(perf_50, main="ROC", colorize=T)
#plot(perf_100, col=2, add=T)
#plot(perf_150, main="ROC", colorize=T)
#plot(perf_250, col=4, add=T)
#plot(perf_350, col=5, add=T)

#plot(perf_unstratified, col=2, add=T)

plot(perf_fullset, main="ROC: Full Model vs. Reduced Model", col="black")
plot(perf_50, col="red", add=T)
#legend(1, 1, c('50', '100', '150','250','350','400', 'unstratified'), 1:7)
legend(0.6, 0.6, c('Initial', 'Reduced'), 1:3)


# Another concern:
# Declining prediction power

# Make prediction on the first 100 instances in the testing set
rf.pred_rfe_14_100 <- predict(rf50, rfe_14_testing[1:100,], type="prob")
pred_14_100 <- prediction(rf.pred_rfe_14_100[,2], rfe_14_testing[1:100,]$daily_difference)
perf_14_100 <- performance(pred_14_100, "tpr", "fpr")
plot(perf_14_100, main="ROC: Recent vs. Distant Test Data", col="red")

# Make prediction on the second 100 instances in the testing set
rf.pred_rfe_14_200 <- predict(rf50, rfe_14_testing[101:200,], type="prob")
pred_14_200 <- prediction(rf.pred_rfe_14_200[,2], rfe_14_testing[101:200,]$daily_difference)
perf_14_200 <- performance(pred_14_200, "tpr", "fpr")
plot(perf_14_200, col="green", add=T)

# Make prediction on the third 100 instances in the testing set
rf.pred_rfe_14_300 <- predict(rf50, rfe_14_testing[201:300,], type="prob")
pred_14_300 <- prediction(rf.pred_rfe_14_300[,2], rfe_14_testing[201:300,]$daily_difference)
perf_14_300 <- performance(pred_14_300, "tpr", "fpr")
plot(perf_14_300, col="blue", add=T)

# Make prediction on the fourth 100 instances in the testing set
rf.pred_rfe_14_400 <- predict(rf50, rfe_14_testing[301:400,], type="prob")
pred_14_400 <- prediction(rf.pred_rfe_14_400[,2], rfe_14_testing[301:400,]$daily_difference)
perf_14_400 <- performance(pred_14_400, "tpr", "fpr")
plot(perf_14_400, col="lightblue", add=T)

legend(0.6, 0.6, c("Interval 1", "Interval 2", "Interval 3","Interval 4"), 2:6)

