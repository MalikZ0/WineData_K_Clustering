library(readxl)
library(neuralnet)
library(Metrics)
library(ggplot2)

CW_exchangeData = read_excel('G:\\iit campus\\course\\2 ND YEAR\\2 sem\\Machine Learning and Data mining\\cw\\ExchangeUSD.xlsx')

# scaling functions
normalization = function(x){
  return((x-min(x))/(max(x)-min(x)))
}
unNormalization = function(x, min, max) {
  return( (max - min)*x + min )
}

# select USD/EUR column
CW_exchangeData = CW_exchangeData[,3]
head(CW_exchangeData)


##### Various input vectors up to (t-4) level.
CW_exchangeData_Tl4 = bind_cols(t_4 = lag(CW_exchangeData, 4),
                             t_3 = lag(CW_exchangeData, 3),
                             t_2 = lag(CW_exchangeData, 2),
                             t_1 = lag(CW_exchangeData, 1),
                             t = `CW_exchangeData`)
# NA values removal and structure
CW_exchangeData_Tl4 = CW_exchangeData_Tl4[complete.cases(CW_exchangeData_Tl4),]
colnames(CW_exchangeData_Tl4) = c("InputV1","InputV2","InputV3","InputV4","Output")

# training and testing matrices 
trainMat_Tl4 = CW_exchangeData_Tl4[1:400,]
testMat_Tl4 = CW_exchangeData_Tl4[401:nrow(CW_exchangeData_Tl4),]

# normalizing the i/o metrics
minV_Tl4 = min(trainMat_Tl4)
maxV_Tl4 = max(trainMat_Tl4)
normData_Tl4 = as.data.frame(lapply(CW_exchangeData_Tl4, normalization))

# spliting normalize data into train and test sets
norm_trainMat_Tl4 = normData_Tl4[1:400,]
norm_testMat_Tl4 = normData_Tl4[401:nrow(CW_exchangeData_Tl4),]

# Hyperparameter tuning ---------------------------------------
# 1st model V(1:4) with one hidden layer 
nn4_1 <- neuralnet(Output ~ InputV1 + InputV2 + InputV3 + InputV4, data = norm_trainMat_Tl4,
                   hidden = 5,
                   linear.output = TRUE, act.fct ="logistic")

# Compute predictions on the test set
modelTl4_h1_5 <- neuralnet::compute(nn4_1, norm_testMat_Tl4)
# normalization of model result
predictionTl4_5 <- unNormalization(modelTl4_h1_5$net.result, minV_Tl4, maxV_Tl4)

# Testing performance 
(rmseTl4_h1_5 = rmse(testMat_Tl4$Output, predictionTl4_5))
(maeTl4_h1_5 = mae(testMat_Tl4$Output, predictionTl4_5))
(mapeTl4_h1_5 = mape(testMat_Tl4$Output, predictionTl4_5))
(smapeTl4_h1_5 = smape(testMat_Tl4$Output, predictionTl4_5))



# 2nd model V(1:4) with two hidden layer
nn4_2 <- neuralnet(Output ~ InputV1 + InputV2 + InputV3 + InputV4, data = norm_trainMat_Tl4,
                   hidden = c(5,2),
                   linear.output = TRUE, act.fct ="logistic")

# Compute predictions on the test set
modelTl4_h2_7 <- neuralnet::compute(nn4_2, norm_testMat_Tl4)
predictionTl4_7 <- unNormalization(modelTl4_h2_7$net.result, minV_Tl4, maxV_Tl4)

(rmseTl4_h2_7 = rmse(testMat_Tl4$Output, predictionTl4_7))
(maeTl4_h2_7 = mae(testMat_Tl4$Output, predictionTl4_7))
(mapeTl4_h2_7 = mape(testMat_Tl4$Output, predictionTl4_7))
(smapeTl4_h2_7 = smape(testMat_Tl4$Output, predictionTl4_7))

# 3rd model V(1:4) with two hidden layer
nn4_3 <- neuralnet(Output ~ InputV1 + InputV2 + InputV3 + InputV4, data = norm_trainMat_Tl4,
                   hidden = c(10,7),
                   linear.output = TRUE, act.fct ="logistic")

# Compute predictions on the test set
modelTl4_h2_17 <- neuralnet::compute(nn4_3, norm_testMat_Tl4)
predictionTl4_17 <- unNormalization(modelTl4_h2_17$net.result, minV_Tl4, maxV_Tl4)

(rmseTl4_h2_17 = rmse(testMat_Tl4$Output, predictionTl4_17))
(maeTl4_h2_17 = mae(testMat_Tl4$Output, predictionTl4_17))
(mapeTl4_h2_17 = mape(testMat_Tl4$Output, predictionTl4_17))
(smapeTl4_h2_17 = smape(testMat_Tl4$Output, predictionTl4_17))


##### Various input vectors up to (t-3) level.
CW_exchangeData_Tl3 = bind_cols( t_3 = lag(CW_exchangeData, 3),
                              t_2 = lag(CW_exchangeData, 2),
                              t_1 = lag(CW_exchangeData, 1),
                              t = `CW_exchangeData`)
# NA values removal and structure
CW_exchangeData_Tl3 = CW_exchangeData_Tl3[complete.cases(CW_exchangeData_Tl3),]
colnames(CW_exchangeData_Tl3) = c("InputV1","InputV2","InputV3","Output")

# training and testing matrices 
trainMat_Tl3 = CW_exchangeData_Tl3[1:400,] 
head(trainMat_Tl3)
testMat_Tl3 = CW_exchangeData_Tl3[401:nrow(CW_exchangeData_Tl3),]

# normalizing the i/o metrics
minV_Tl3 = min(trainMat_Tl3)  
maxV_Tl3 = max(trainMat_Tl3)
normData_Tl3 = as.data.frame(lapply(CW_exchangeData_Tl3, normalization))

# spliting normalize data into train and test sets
norm_trainMat_Tl3 = normData_Tl3[1:400,]
head(norm_trainMat_Tl3)
norm_testMat_Tl3 = normData_Tl3[401:nrow(CW_exchangeData_Tl3),] 
head(norm_testMat_Tl3)

# Hyperparameter tuning ---------------------------------------
# 1st model V(1:3) with one hidden layer
nn3_1 <- neuralnet(Output ~ InputV1 + InputV2 + InputV3, data = norm_trainMat_Tl3,
                   hidden = 5,
                   linear.output = TRUE, act.fct ="logistic")

# Compute predictions on the test set
modelTl3_h1_5 <- neuralnet::compute(nn3_1, norm_testMat_Tl3)
predictionTl3_5 <- unNormalization(modelTl3_h1_5$net.result, minV_Tl3, maxV_Tl3)

# Testing performance 
(rmseTl3_h1_5 = rmse(testMat_Tl3$Output, predictionTl3_5))
(maeTl3_h1_5 = mae(testMat_Tl3$Output, predictionTl3_5))
(mapeTl3_h1_5 = mape(testMat_Tl3$Output, predictionTl3_5))
(smapeTl3_h1_5 = smape(testMat_Tl3$Output, predictionTl3_5))

# 2nd model V(1:3) with two hidden layer
nn3_2 <- neuralnet(Output ~ InputV1 + InputV2 + InputV3, data = norm_trainMat_Tl3,
                   hidden = c(5,2),
                   linear.output = TRUE, act.fct ="logistic")

# Compute predictions on the test set
modelTl3_h2_7 <- neuralnet::compute(nn3_2, norm_testMat_Tl3)
predictionTl3_7 <- unNormalization(modelTl3_h2_7$net.result, minV_Tl3, maxV_Tl3)

# Testing performance 
(rmseTl3_h2_7 = rmse(testMat_Tl3$Output, predictionTl3_7))
(maeTl3_h2_7 = mae(testMat_Tl3$Output, predictionTl3_7))
(mapeTl3_h2_7 = mape(testMat_Tl3$Output, predictionTl3_7))
(smapeTl3_h2_7 = smape(testMat_Tl3$Output, predictionTl3_7))

# 3rd model V(1:3) with two hidden layer
nn3_3 <- neuralnet(Output ~ InputV1 + InputV2 + InputV3, data = norm_trainMat_Tl3,
                   hidden = c(10,7),
                   linear.output = TRUE, act.fct ="logistic")

# Compute predictions on the test set
modelTl3_h2_17 <- neuralnet::compute(nn3_3, norm_testMat_Tl3)
predictionTl3_17 <- unNormalization(modelTl3_h2_17$net.result, minV_Tl3, maxV_Tl3)

# Testing performance 
(rmseTl3_h2_17 = rmse(testMat_Tl3$Output, predictionTl3_17))
(maeTl3_h2_17 = mae(testMat_Tl3$Output, predictionTl3_17))
(mapeTl3_h2_17 = mape(testMat_Tl3$Output, predictionTl3_17))
(smapeTl3_h2_17 = smape(testMat_Tl3$Output, predictionTl3_17))


##### Various input vectors up to (t-2) level.
CW_exchangeData_Tl2 = bind_cols(t_2 = lag(CW_exchangeData, 2),
                             t_1 = lag(CW_exchangeData, 1),
                             t = `CW_exchangeData`)
# NA values removal and structure
CW_exchangeData_Tl2 = CW_exchangeData_Tl2[complete.cases(CW_exchangeData_Tl2),]
colnames(CW_exchangeData_Tl2) = c("InputV1","InputV2","Output")

# training and testing matrices
trainMat_Tl2 = CW_exchangeData_Tl2[1:400,]
head(trainMat_Tl2)
testMat_Tl2 = CW_exchangeData_Tl2[401:nrow(CW_exchangeData_Tl2),]

# normalizing the i/o metrics
minV_Tl2 = min(trainMat_Tl2)  
maxV_Tl2 = max(trainMat_Tl2)
normData_Tl2 = as.data.frame(lapply(CW_exchangeData_Tl2, normalization)) 

# spliting normalize data into train and test sets
norm_trainMat_Tl2 = normData_Tl2[1:400,]   
head(norm_trainMat_Tl2)
norm_testMat_Tl2 = normData_Tl2[401:nrow(CW_exchangeData_Tl2),]
head(norm_testMat_Tl2)

# Hyperparameter tuning ---------------------------------------
# 1st model V(1:2) with one hidden layer
nn2_1 <- neuralnet(Output ~ InputV1 + InputV2 , data = norm_trainMat_Tl2,
                   hidden = 5,
                   linear.output = TRUE, act.fct ="logistic")

# Compute predictions on the test set
modelTl2_h1_5 <- neuralnet::compute(nn2_1, norm_testMat_Tl2)
predictionTl2_5 <- unNormalization(modelTl2_h1_5$net.result, minV_Tl2, maxV_Tl2)

# Testing performance 
(rmseTl2_h1_5 = rmse(testMat_Tl2$Output, predictionTl2_5))
(maeTl2_h1_5 = mae(testMat_Tl2$Output, predictionTl2_5))
(mapeTl2_h1_5 = mape(testMat_Tl2$Output, predictionTl2_5))
(smapeTl2_h1_5 = smape(testMat_Tl2$Output, predictionTl2_5))

# 2nd model V(1:2) with two hidden layer
nn2_2 <- neuralnet(Output ~ InputV1 + InputV2, data = norm_trainMat_Tl2,
                   hidden = c(5,2),
                   linear.output = TRUE, act.fct ="logistic")

# Compute predictions on the test set
modelTl2_h2_7 <- neuralnet::compute(nn2_2, norm_testMat_Tl2)
predictionTl2_7 <- unNormalization(modelTl2_h2_7$net.result, minV_Tl2, maxV_Tl2)

# Testing performance 
(rmseTl2_h2_7 = rmse(testMat_Tl2$Output, predictionTl2_7))
(maeTl2_h2_7 = mae(testMat_Tl2$Output, predictionTl2_7))
(mapeTl2_h2_7 = mape(testMat_Tl2$Output, predictionTl2_7))
(smapeTl2_h2_7 = smape(testMat_Tl2$Output, predictionTl2_7))

# 3rd model V(1:2) with two hidden layer
nn2_3 <- neuralnet(Output ~ InputV1 + InputV2, data = norm_trainMat_Tl2,
                   hidden = c(10,7),
                   linear.output = TRUE, act.fct ="logistic")

# Compute predictions on the test set
modelTl2_h2_17 <- neuralnet::compute(nn2_3, norm_testMat_Tl2)
predictionTl2_17 <- unNormalization(modelTl2_h2_17$net.result, minV_Tl2, maxV_Tl2)

# Testing performance 
(rmseTl2_h2_17 = rmse(testMat_Tl2$Output, predictionTl2_17))
(maeTl2_h2_17 = mae(testMat_Tl2$Output, predictionTl2_17))
(mapeTl2_h2_17 = mape(testMat_Tl2$Output, predictionTl2_17))
(smapeTl2_h2_17 = smape(testMat_Tl2$Output, predictionTl2_17))

##### Various input vectors up to (t-1) level
CW_exchangeData_Tl1 = bind_cols(t_1 = lag(CW_exchangeData, 1),
                             t = `CW_exchangeData`)
# NA values removal and structure
CW_exchangeData_Tl1 = CW_exchangeData_Tl1[complete.cases(CW_exchangeData_Tl1),]
colnames(CW_exchangeData_Tl1) = c("InputV1","Output")

# training and testing matrices 
trainMat_Tl1 = CW_exchangeData_Tl1[1:400,]
head(trainMat_Tl2)
testMat_Tl1 = CW_exchangeData_Tl1[401:nrow(CW_exchangeData_Tl1),] 

# normalizing the i/o metrics
minV_Tl1 = min(trainMat_Tl1) 
maxV_Tl1 = max(trainMat_Tl1)
normData_Tl1 = as.data.frame(lapply(CW_exchangeData_Tl1, normalization)) 

# spliting normalize data into train and test sets
norm_trainMat_Tl1 = normData_Tl1[1:400,] 
head(norm_trainMat_Tl1)
norm_testMat_Tl1 = normData_Tl1[401:nrow(CW_exchangeData_Tl1),]
head(norm_testMat_Tl1)

# 1st model V(1:1) with one hidden layer
nn1_1 <- neuralnet(Output ~ InputV1 , data = norm_trainMat_Tl1,
                   hidden = 5,
                   linear.output = TRUE, act.fct ="logistic")

# Compute predictions on the test set
modelTl1_h1_5 <- neuralnet::compute(nn1_1, norm_testMat_Tl1)
predictionTl1_5 <- unNormalization(modelTl1_h1_5$net.result, minV_Tl1, maxV_Tl1)

# Testing performance 
(rmseTl1_h1_5 = rmse(testMat_Tl1$Output, predictionTl1_5))
(maeTl1_h1_5 = mae(testMat_Tl1$Output, predictionTl1_5))
(mapeTl1_h1_5 = mape(testMat_Tl1$Output, predictionTl1_5))
(smapeTl1_h1_5 = smape(testMat_Tl1$Output, predictionTl1_5))

# 2nd model V(1:1) with two hidden layer
nn1_2 <- neuralnet(Output ~ InputV1, data = norm_trainMat_Tl1,
                   hidden = c(5,2),
                   linear.output = TRUE, act.fct ="logistic")

# Compute predictions on the test set
modelTl1_h2_7 <- neuralnet::compute(nn1_2, norm_testMat_Tl1)
predictionTl1_7 <- unNormalization(modelTl1_h2_7$net.result, minV_Tl1, maxV_Tl1)

# Testing performance 
(rmseTl1_h2_7 = rmse(testMat_Tl1$Output, predictionTl1_7))
(maeTl1_h2_7 = mae(testMat_Tl1$Output, predictionTl1_7))
(mapeTl1_h2_7 = mape(testMat_Tl1$Output, predictionTl1_7))
(smapeTl1_h2_7 = smape(testMat_Tl1$Output, predictionTl1_7))

# 3rd model V(1:1) with two hidden layer
nn1_3 <- neuralnet(Output ~ InputV1, data = norm_trainMat_Tl1,
                   hidden = c(10,7),
                   linear.output = TRUE, act.fct ="logistic")

# Compute predictions on the test set
modelTl1_h2_17 <- neuralnet::compute(nn1_3, norm_testMat_Tl1)
predictionTl1_17 <- unNormalization(modelTl1_h2_17$net.result, minV_Tl1, maxV_Tl1)

# Testing performance 
(rmseTl1_h2_17 = rmse(testMat_Tl1$Output, predictionTl1_17))
(maeTl1_h2_17 = mae(testMat_Tl1$Output, predictionTl1_17))
(mapeTl1_h2_17 = mape(testMat_Tl1$Output, predictionTl1_17))
(smapeTl1_h2_17 = smape(testMat_Tl1$Output, predictionTl1_17))


# all performance evaluation 
rmse_values = c(rmseTl1_h1_5, rmseTl2_h1_5, rmseTl3_h1_5, rmseTl4_h1_5, 
            rmseTl1_h2_7, rmseTl2_h2_7, rmseTl3_h2_7, rmseTl4_h2_7, 
            rmseTl1_h2_17, rmseTl2_h2_17, rmseTl3_h2_17, rmseTl4_h2_17)

mae_values = c(maeTl1_h1_5, maeTl2_h1_5, maeTl3_h1_5, maeTl4_h1_5, 
           maeTl1_h2_7, maeTl2_h2_7, maeTl3_h2_7, maeTl4_h2_7, 
           maeTl1_h2_17,maeTl2_h2_17, maeTl3_h2_17, maeTl4_h2_17)

mape_values = c(mapeTl1_h1_5, mapeTl2_h1_5, mapeTl3_h1_5, mapeTl4_h1_5, 
            mapeTl1_h2_7, mapeTl2_h2_7, mapeTl3_h2_7, mapeTl4_h2_7, 
            mapeTl1_h2_17,mapeTl2_h2_17, mapeTl3_h2_17, mapeTl4_h2_17)

smape_values = c(smapeTl1_h1_5, smapeTl2_h1_5, smapeTl3_h1_5, smapeTl4_h1_5, 
             smapeTl1_h2_7, smapeTl2_h2_7, smapeTl3_h2_7, smapeTl4_h2_7, 
             smapeTl1_h2_17,smapeTl2_h2_17, smapeTl3_h2_17, smapeTl4_h2_17)
# all plots of nueral net models
plot(nn1_1)
plot(nn1_2)
plot(nn1_3)
plot(nn2_1)
plot(nn2_2)
plot(nn2_3)
plot(nn3_1)
plot(nn3_2)
plot(nn3_3)
plot(nn4_1)
plot(nn4_2)
plot(nn4_3)
# Comparison table
comparison_Table = data.frame(Input_nodes = c("1","2","3","4","1","2","3","4","1","2","3","4"),
                              Hidden_layers = c("1","1","1","1","2","2","2","2","2","2","2","2"),
                              HIdden_nodes = c("5","5","5","5", "5-2", "5-2", "5-2", "5-2", "10-7", "10-7", "10-7", "10-7"),
                              RMSE = rmse_values,
                              MAE = mae_values,
                              MAPE = mape_values,
                              SMAPE = smape_values)
comparison_Table


# Evaluate the most accurate model
# plot for the selected model
plot(nn2_1)
plot(nn3_3)
# Actual and Predicted Exchange Rates Time plot for the model
exchange_ratesPlot <- data.frame(
  DateSeq = seq(as.Date("2013-05-15"), by = "day", length.out = length(testMat_Tl3$Output)),
  ActualRate = testMat_Tl3$Output,
  PredictedRate = predictionTl3_17
)

ggplot(exchange_ratesPlot, aes(x = DateSeq)) +
  geom_line(aes(y = ActualRate, color = "Actual")) +
  geom_line(aes(y = PredictedRate, color = "Predicted"), linetype = "dashed") +
  labs(x = "Date", y = "Exchange Rate", color = "Series") +
  ggtitle("Time Plot of Exchange Rates")

# Actual and Predicted Exchange Rates scatter plot for the model
exchange_ScatPlot <- data.frame(
  ActualRate = testMat_Tl3$Output,
  PredictedRate =  predictionTl3_17
)

ggplot(exchange_ScatPlot, aes(x = ActualRate, y = PredictedRate)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(x = "Actual Exchange Rate", y = "Predicted ") +
  ggtitle("Scatter Plot of Exchange Rates")