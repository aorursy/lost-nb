# Kaggle-competititon "Avito Context Ad Clicks"
# See https://www.kaggle.com/c/avito-context-ad-clicks

# In order to run this script on Kaggle-scripts I had to limit the number of entries to read
# from the database as well as to decrease the sample-size. With the full dataset from the database as well
# as a sample of 10 millions entries I got a 0.05104 on the public leaderboard

library("data.table")
library("RSQLite")
library("caret")

# ----- Prepare database -------------------------------------------------------

db <- dbConnect(SQLite(), dbname="../input/database.sqlite")
dbListTables(db)

# ----- Utitlies ---------------------------------------------------------------

# Define constants to improve readability of large number
thousand <- 1000
million  <- thousand * thousand 
billion  <- thousand * million

# Runs the query, fetches the given number of entries and returns a
# data.table
fetch  <- function(db, query, n = -1) {
  result <- dbSendQuery(db, query)
  data <- dbFetch(result, n)
  dbClearResult(result)
  return(as.data.table(data))
}

# Loss-function to evaluate result
# See https://www.kaggle.com/c/avito-context-ad-clicks/details/evaluation
logloss <- function(y, yHat){
  
  threshold <- 10^(-15)
  yHat <- pmax(pmin(yHat, 1-threshold), threshold)
  
  loss <- -mean(y*log(yHat) + (1-y)*log(1-yHat))
  
  return(loss)
}

# ----- Simple Machine Learning ------------------------------------------------

# Select contextual Ads (OnjectType=3), results in 190.157.735 entries
# Warning: Takes a few minutes
trainSearchStreamContextual <- fetch(db, "select HistCTR, IsClick from trainSearchStream where ObjectType=3", 10 * million)
m <- nrow(trainSearchStreamContextual)

# Create stratified sample 
sampleSize <- 1 * million #100 * million
sampleRatio <- sampleSize / m
sampleIndex <- createDataPartition(trainSearchStreamContextual$IsClick, p = sampleRatio, list=FALSE)
trainSearchStreamContextualSample <- trainSearchStreamContextual[as.vector(sampleIndex), ]

# Compare click-ratio in full set and sample to verify stratification
print(paste("Clickratio full dataset:", sum(trainSearchStreamContextual$IsClick)/m))
print(paste("Clickratio sample:", sum(trainSearchStreamContextualSample$IsClick)/sampleSize))

# Create stratified random split ...
trainSampleIndex <- createDataPartition(y = trainSearchStreamContextualSample$IsClick, p = .80, list = FALSE)

# ... and partition data-set into train- and validation-set
trainSearchStreamContextualTrainSample <- trainSearchStreamContextualSample[as.vector(trainSampleIndex),]
trainSearchStreamContextualValidationSample <- trainSearchStreamContextualSample[-as.vector(trainSampleIndex),]

# Build a logistic regression ...
model <- glm(IsClick ~ HistCTR, data = trainSearchStreamContextualTrainSample, family="binomial")

# Check that regression-coefficients have significant impact
summary(model)

# ... and predict data on validation data-set
prediction <- predict(model, trainSearchStreamContextualValidationSample, type="response")
print(logloss(trainSearchStreamContextualValidationSample$IsClick, prediction))

# ----- Predict submission dataset ---------------------------------------------

testSearchStreamContextual <- fetch(db, "select TestId, HistCTR from testSearchStream where ObjectType=3")
prediction <- predict(model, testSearchStreamContextual, type="response")

submissionData <- data.frame(ID=testSearchStreamContextual$TestId, IsClick=prediction)
submissionFile <- paste0("glm", format(Sys.time(), "%Y-%m-%d-%H:%M:%S"), ".csv")
#write.csv(submissionData, submissionFile, sep=",", dec=".", col.names=TRUE, row.names=FALSE)

# ----- Clean up ---------------------------------------------------------------

dbDisconnect(db)


