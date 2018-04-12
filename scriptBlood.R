cat("\014") 
rm(list=ls())

# LOADING DATA
library(rstudioapi)    
library(ggplot2)
library(pscl)
library(mvoutlier)
library(MVN)
library(caret)
library(corrplot)
library(LiblineaR)
library(RWeka)
library(rpart.plot)
library(randomForest)

path = rstudioapi::getActiveDocumentContext()$path
path = sub(basename(path), '', path)

setwd(path)
blood = read.csv(file = file.choose(), stringsAsFactors = F)
#summary(blood)

############################
# CLEANING THE DATABASE
##########################

blood = blood[,-1]
names(blood) = c("MonthsLastDonation", "NoDonations", "TotalVolume", "MonthsFirstDonation", "MadeDonation")
## Remove the variable TotalVolume since the volume is the number of donations * 250cc
blood = blood[,-3]

##########
#PLOTS
##########

pairs(blood)
boxplot(blood$MonthsLastDonation~blood$MadeDonation)
boxplot(blood$NoDonations~blood$MadeDonation)
boxplot(blood$MonthsFirstDonation~blood$MadeDonation)

ggplot(blood, aes(MonthsLastDonation, MonthsFirstDonation)) +
  geom_point(aes(colour = factor(MadeDonation)))

ggplot(blood, aes(MonthsLastDonation, NoDonations)) +
  geom_point(aes(colour = factor(MadeDonation)))

ggplot(blood, aes(NoDonations, MonthsFirstDonation)) +
  geom_point(aes(colour = factor(MadeDonation)))

featurePlot(x = blood[, 1:3], 
            y = as.factor(blood$MadeDonation), 
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))

featurePlot(x = blood[, 1:3], 
            y = as.factor(blood$MadeDonation), 
            plot = "ellipse",
            ## Add a key at the top
            auto.key = list(columns = 3))

featurePlot(x = blood[, 1:3], 
           y = as.factor(blood$MadeDonation),
           plot = "density", 
           ## Pass in options to xyplot() to 
           ## make it prettier
           scales = list(x = list(relation="free"), 
                         y = list(relation="free")), 
           adjust = 1.5, 
           pch = "|", 
           layout = c(3, 1), 
           auto.key = list(columns = 3))

cor <- cor(blood[,-4])
corrplot::corrplot(cor)

##################
# SPLITING DATA
################

set.seed(123)
trainIndex <- createDataPartition(blood$MadeDonation, p = .8, 
                                  list = FALSE, 
                                  times = 1)

bloodTrain <- blood[trainIndex,]
bloodTest <- blood[-trainIndex,]


## it seems there's a outliers in the data. 
## Proceed to outlier detection.

mod <- glm(MadeDonation~MonthsLastDonation+NoDonations+MonthsFirstDonation, 
           family=binomial(link=logit), data = bloodTrain)

cooksd <- cooks.distance(mod)

plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="red")  # add labels

#summary(mod)
pR2(mod)

influential <- as.numeric(names(cooksd)[(cooksd > 4*mean(cooksd, na.rm=T))]) 
bloodTrain <- bloodTrain[-influential,]


mod2 <- glm(MadeDonation~MonthsLastDonation+NoDonations+MonthsFirstDonation, 
           family=binomial(link=logit), data = bloodTrain)

pR2(mod2)

ggplot(bloodTrain, aes(MonthsLastDonation, MonthsFirstDonation)) +
  geom_point(aes(colour = factor(MadeDonation)))

ggplot(bloodTrain, aes(MonthsLastDonation, NoDonations)) +
  geom_point(aes(colour = factor(MadeDonation)))

ggplot(bloodTrain, aes(NoDonations, MonthsFirstDonation)) +
  geom_point(aes(colour = factor(MadeDonation)))


aq<- aq.plot(trainTransformed[-4])
#chisq.plot(blood[-4])


outliers<- c(1,9,387,389,264,398,388,93,5,391,524,4,517,516,189,573)

# Calculate Mahalanobis with predictor variables
df2 <- bloodTrain[, -4]    
m_dist <- mahalanobis(df2, colMeans(df2), cov(df2))
df2$MD <- round(m_dist, 1)

# Binary Outlier Variable
df2$outlier <- "No"
df2$outlier[df2$MD > 15] <- "Yes"    # Threshold set to 20

ggplot(df2, aes(NoDonations, MonthsFirstDonation)) +
  geom_point(aes(colour = factor(outlier)))
ggplot(df2, aes(NoDonations, MonthsLastDonation)) +
  geom_point(aes(colour = factor(outlier)))
ggplot(df2, aes(MonthsFirstDonation, MonthsLastDonation)) +
  geom_point(aes(colour = factor(outlier)))



## the outlier from the Cooks distances method seems better than 
## Mahalanobis, so bllod 2 is the new dataset to work on

##################
# SCALING DATA
#################

preProcValues <- preProcess(bloodTrain[-4], method = c("center", "scale"))

trainTransformed <- predict(preProcValues, bloodTrain)
testTransformed <- predict(preProcValues, bloodTest)

########################
# LOGISTIC REGRESSION
########################

#Linear Regression between Y and X

lm1 = lm(MadeDonation~MonthsLastDonation+NoDonations+MonthsFirstDonation, data = trainTransformed)
summary(lm1)

#NON linear regression since the response variable is categorical. Using Logistic Regression.

glm1 = glm(MadeDonation~MonthsLastDonation+NoDonations+MonthsFirstDonation, family=binomial(link=logit), data = trainTransformed)
summary(glm1)

pR2(glm1)

preds<- ifelse(predict(glm1, newdata = testTransformed)>.5,1,0)


confusionMatrix(data = preds, reference = testTransformed$MadeDonation, positive = '1')$overall[1]

acurracy <- numeric(8)
Y <- trainTransformed$MadeDonation
X<- trainTransformed[,1:3]
X2<- testTransformed[,1:3]
Y2<- testTransformed$MadeDonation

polynomial <- function(X, p){
  new_data=c()
  temp1 <- X
  temp2 <- cbind(X^2, do.call(cbind,combn(colnames(X), 2, 
                FUN= function(x) list(X[x[1]]*X[x[2]]))))
  colnames(temp2)[-(seq_len(ncol(X)))] <-  combn(colnames(X), 2, 
                                                 FUN = paste, collapse=":")
  new_data <- cbind(temp1,  temp2[, !names(temp2) %in% names(X)])
  
  for(i in 2:p){
      temp1= as.data.frame(X^p)
      colnames(temp1)<-paste0(colnames(temp1), paste0("^",i))
      temp2 <- cbind(temp1^2, do.call(cbind,combn(colnames(temp1), 2, 
                FUN= function(x) list(temp1[x[1]]*temp1[x[2]]))))
      colnames(temp2)[-(seq_len(ncol(temp1)))] <-  combn(colnames(temp1), 2, 
                                                     FUN = paste, collapse=":")
      new_data=cbind(new_data,temp1, temp2[, !names(temp2) %in% names(temp1)])
  }
  
  #colnames(new_data)=paste0("V",1:ncol(new_data))
  return(as.data.frame(new_data))
}

for (i in 2:8){
  #X <- as.data.frame(do.call(poly, c(lapply(1:(ncol(X)), function(x) X[,x]), degree=i, raw=T)))
  #X2 <- as.data.frame(do.call(poly, c(lapply(1:(ncol(X2)), function(x) X2[,x]), degree=i, raw=T)))
  
  #colnames(X)<- as.character(c(1:ncol(X)))
  #colnames(X2)<- as.character(c(1:ncol(X2)))
  
  X = polynomial(X,i)
  X2 = polynomial(X2,i)
  
  data<-cbind(X,Y)
  
  glm2 <- glm(Y~., family=binomial(link=logit), data = data)
  #X<-as.data.frame(X)
  #X2<-as.data.frame(X2)
  #colnames(X2)<- names(X)
  preds<- ifelse(predict(glm2, newdata = X2)>.5,1,0)
  acurracy[i] <- confusionMatrix(data = preds, reference = Y2, positive = '1')$overall[1]
}

plot(acurracy)

## 3 polynomial has the best accuracy

X<- trainTransformed[,1:3]
X2<- testTransformed[,1:3]

X = polynomial(X,3)
X2 = polynomial(X2,3)

#colnames(X)<-c("MonthsLastDonation","NoDonations","MonthsFirstDonation",
#               "MonthsLastDonation2","NoDonations2","MonthsFirstDonation2")

#colnames(X2)<-c("MonthsLastDonation","NoDonations","MonthsFirstDonation",
#               "MonthsLastDonation2","NoDonations2","MonthsFirstDonation2")


##################################################
# REGULARIZED LOGISTIC REGRESSION WITH 2 POLYNOMIAL
###################################################


# Find the best model with the best cost parameter via 10-fold cross-validations
tryTypes=c(0:7)
tryCosts=c(1000,1,0.001)
bestCost=NA
bestAcc=0
bestType=NA

for(ty in tryTypes){
  for(co in tryCosts){
    acc=LiblineaR(data=X,target=Y,type=ty,cost=co,bias=1,cross=5,verbose=FALSE)
    cat("Results for C=",co," : ",acc," accuracy.\n",sep="")
    if(acc>bestAcc){
      bestCost=co
      bestAcc=acc
      bestType=ty
    }
  }
}

cat("Best model type is:",bestType,"\n")
cat("Best cost is:",bestCost,"\n")
cat("Best accuracy is:",bestAcc,"\n")

## Best model type is: 6 
## Best cost is: 1 
## Best accuracy is: 0.8008949
bestCost=1
bestType=6


modLin=LiblineaR(data=X,target=Y,type=bestType,cost=bestCost,bias=1,verbose=FALSE)

pr=FALSE
if(bestType==0 || bestType==7) pr=TRUE

p=predict(modLin,X2,proba=pr,decisionValues=TRUE)

res=table(p$predictions,Y2)
print(res)

confusionMatrix(data = p$predictions, reference = Y2, positive = '1')$overall[1]

###########################
# DECISION TREES
########################

fitControl <- trainControl(method = "cv", number = 2)

rpartFit <- train(as.character(MadeDonation) ~., data=bloodTrain, method = "rpart", trControl = fitControl)

probsTest <- predict(rpartFit, bloodTest, type = "prob")
pred      <- factor( ifelse(probsTest[, "1"] > 0.5, "1", "0") )
confusionMatrix(pred, bloodTest$MadeDonation)

rpart.plot(rpartFit$finalModel)

#########################
# DECISION TREES
#####################

bloodTrain$MadeDonation<-as.character(bloodTrain$MadeDonation)
blood$MadeDonation<-as.character(blood$MadeDonation)

blood.rf=randomForest(MadeDonation ~ . , data = bloodTrain)
blood.rf

plot(blood.rf)




library(RWeka)
blood[,5] = as.factor(blood[,5])
blood<-blood[,-3]  
smp_size <- floor(0.75 * nrow(blood))  ## set the seed to make your partition reproductible 
set.seed(123) 
train_ind <- sample(seq_len(nrow(blood)), size = smp_size)  
train <- blood[train_ind, ] 
test <- blood[-train_ind, ]
fit <- J48(MadeDonation~MonthsLastDonation+NoDonations+MonthsFirstDonation, data=trainTransformed)
summary(fit)

library(partykit)
plot(fit)

pr = subset(train, MonthsLastDonation <=7 & NoDonations<=18 & MonthsFirstDonation>49)
summary(pr)
predict(fit, pr)

samNeg = blood[sample(row.names(subset(blood, MadeDonation == 0)),100,replace = T ),]
samPos = blood[sample(row.names(subset(blood, MadeDonation == 1)),100,replace = T ),]

train_b = samNeg
train_b = rbind(train_b, samPos)
summary(train_b)
fit <- J48(MadeDonation~MonthsLastDonation+NoDonations+MonthsFirstDonation, data=train_b)
summary(fit)
