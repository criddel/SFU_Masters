## Some code adapted from STAT 652 Fall 2020 - T. Loughin
## Final Project - Predicting fish populations using a number of statistical models


#Input Dataset
data = read.csv('fish.csv')

set.seed(436564577)
library(caret)
library(glmnet)
library(MASS) 
library(randomForest)
library(nnet)
library(foreach)

x11(h=7,w=10,pointsize=8)
data$Species = as.factor(data$Species)
dmy = dummyVars("~.", data=data)
data = data.frame(predict(dmy, newdata = data))
#data = data[,-1]
perm <- sample(x=nrow(data))
df_train <- data[which(perm <= 1000),]
df_test <- data[which(perm>1000),]
#head(df_train,100)
rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}
data
get.folds = function(n, K) {
  ### Get the appropriate number of fold labels
  n.fold = ceiling(n / K) # Number of observations per fold (rounded up)
  fold.ids.raw = rep(1:K, times = n.fold) # Generate extra labels
  fold.ids = fold.ids.raw[1:n] # Keep only the correct number of labels
  
  ### Shuffle the fold labels
  folds.rand = fold.ids[sample.int(n)]
  
  return(folds.rand)
}

#Set up matrix of test scores
library(mgcv)
library(gbm)
n = nrow(df_train)
R = 5
V = 10
folds = get.folds(n,V)
MSPEs.cv = matrix(NA, nrow=V, ncol=8)
colnames(MSPEs.cv) = c("LM", "GBM", "GAM", "NN", "Ridge", "LASSO-M", "LASSO-1", "RF")
MSPEs.cv[,1] = 0 
MSPEs.cv[,2] = 0 
MSPEs.cv[,3] = 0
MSPEs.cv[,4] = 0
MSPEs.cv[,5] = 0
MSPEs.cv[,6] = 0
MSPEs.cv[,7] = 0
MSPEs.cv[,8] = 0
MSPEs.cv[,9] = 0


# R runs 
# V folds for cross-validation
for (r in 1:R){
  for (v in 1:V){
    
    #Train data
    y.1 <- as.matrix(df_train[folds!=v,22])
    x.1.unscaled <- as.matrix(df_train[folds!=v,-22]) # Original data set 1
    x.1 <- rescale(x.1.unscaled, x.1.unscaled) #scaled data set 1
    xy.1 <- data.frame(x.1, y.1)
    
    #Test
    y.2 <- df_train[folds==v,22]
    y.2.mx <- as.matrix(y.2)
    x.2.unscaled <- as.matrix(df_train[folds==v,-22]) # Original data set 1
    x.2 <- rescale(x.2.unscaled, x.1.unscaled) #scaled data set 1
    xy.2 <- data.frame(x.2, y.2)
    
    #Linear Regression
    lm = lm(y.1 ~ ., data=xy.1)
    summary(lm)
    
    #Lasso Regression
    cv.lasso.1 <- cv.glmnet(y=y.1, x= x.1, family="gaussian")
    
    prb.6_.1 <- gbm(data=xy.1, y.1 ~ ., distribution="gaussian", 
                    n.trees=500, interaction.depth=25, shrinkage=0.005, 
                    bag.fraction=0.8)
    summary(prb.6_.1)
    
    #GAM
    gam.all <- gam(data=xy.1,
                   formula=y.1 ~  Species.Campostoma_anomalum + Species.Campostoma_oligolepis + Species.Cottus_carolinae + Species.Cyprinella_analostoma + Species.Cyprinella_spiloptera +
                   Species.Cyprinella_venusta + Species.Esox_americanus + Species.Etheostoma_blennioides + Species.Etheostoma_caeruleum + Species.Etheostoma_flabellare + Species.Etheostoma_olmstedi + 
                   Species.Fundulus_olivaceus + Species.Lepistoseus_oculatus + Species.Lepistoseus_osseus + Species.Moxostoma_duquesnei + Species.Moxostoma_erythrurum + Species.Noturus_exilis + 
                   Species.Percina_nigrofasciata + Species.Percina_sciera +
                     s(Longitude)+s(Latitude) + s(Ann_mean_temp) + s(Mean_diurnal_range) + s(temp_seasonality) + s(Max_temp) + s(Min_temp) + s(temp_annual_range)
                   + s(Annual_precip) + s(wettest_month) + s(driest_month) +  s(Precip_seasonality), 
                   family=gaussian(link=identity)) 
    summary(gam.all)
    
    #Neural Network
    nn.final <- nnet(y=y.1, x=x.1, linout=TRUE, size=4, decay=0.001, maxit=200, trace=TRUE)
    
    #Random Forest
    pro.rf <- randomForest(data=xy.1, y.1 ~ ., 
                           importance=TRUE, ntree=500, mtry=1, nodesize=25,
                           keep.forest=TRUE)
    
    preds1 = predict(lm, newdata=xy.2)
    score1 = mean((preds1 - y.2)^2)
    
    preds2 = predict(prb.6_.1, newdata=xy.2, n.trees=500)
    score2 = mean((preds2 - y.2)^2)
    
    preds3 = predict(gam.all, newdata=xy.2)
    score3 = mean((preds3 - y.2)^2)
    
    pred.nn = predict(nn.final, newdata=xy.2)
    score4 = mean((pred.nn - y.2)^2)
    
    #Ridge Regression
    ridge1 <- lm.ridge(y.1 ~., lambda = seq(0, 100, .05), data=xy.1)
    
    select(ridge1)
    coef.ri.best1 = coef(ridge1)[which.min(ridge1$GCV),]
    
    pred.ri1 = as.matrix(cbind(1,x.2)) %*% coef.ri.best1
    score5 = mean((y.2-pred.ri1)^2)
    
    pred.6 = predict(cv.lasso.1, newx=x.2, s=cv.lasso.1$lambda.min)
    score6 = mean((pred.6 - y.2.mx)^2)
    
    pred.7 = predict(cv.lasso.1, newx=x.2, s=cv.lasso.1$lambda.1se)
    score7 = mean((pred.7 - y.2.mx)^2)
    pro.rfm2 <- randomForest(data=xy.2, y.2~., ntree=500, 
                            mtry=1, nodesize=25)
    score8 <- mean((predict(pro.rfm2) - y.2)^2)

    length_divisor <- 4  
    iterations <- 100  
    predictions <- foreach(m=1:iterations,.combine=cbind) %do% {  
      training_positions <- sample(nrow(xy.1), size=floor((nrow(xy.1)/length_divisor)))  
      train_pos<-1:nrow(xy.1) %in% training_positions  
      lm_fit <- lm(y.1 ~., data=xy.1)
      predict(lm_fit, newdata=xy.2)
    }  
    predictions<-rowMeans(predictions)  
    score9<-(mean((y.2-predictions)^2)) 
    
    MSPEs.cv[v,1] = MSPEs.cv[v,1] + score1 
    MSPEs.cv[v,2] = MSPEs.cv[v,2] + score2 
    MSPEs.cv[v,3] = MSPEs.cv[v,3] + score3 
    MSPEs.cv[v,4] = MSPEs.cv[v,4] + score4 
    MSPEs.cv[v,5] = MSPEs.cv[v,5] + score5 
    MSPEs.cv[v,6] = MSPEs.cv[v,6] + score6
    MSPEs.cv[v,7] = MSPEs.cv[v,7] + score7
    MSPEs.cv[v,8] = MSPEs.cv[v,8] + score8
    #MSPEs.cv[v,9] = MSPEs.cv[v,9] + score9
  }
}

MSPEs.cv[,1] = MSPEs.cv[,1]/R
MSPEs.cv[,2] = MSPEs.cv[,2]/R
MSPEs.cv[,3] = MSPEs.cv[,3]/R
MSPEs.cv[,4] = MSPEs.cv[,4]/R 
MSPEs.cv[,5] = MSPEs.cv[,5]/R
MSPEs.cv[,6] = MSPEs.cv[,6]/R
MSPEs.cv[,7] = MSPEs.cv[,7]/R
MSPEs.cv[,8] = MSPEs.cv[,8]/R


mean(MSPEs.cv[,1])
mean(MSPEs.cv[,2])
mean(MSPEs.cv[,3])
mean(MSPEs.cv[,4])
mean(MSPEs.cv[,5])
mean(MSPEs.cv[,6])
mean(MSPEs.cv[,7])
mean(MSPEs.cv[,8])

x11(h=7,w=10,pointsize=8)
par(mfrow=c(1,1))
low.s = apply(MSPEs.cv, 1, min)
boxplot(MSPEs.cv, las=2,
        main="MSPE \n Cross Validation")

x11(h=7,w=10,pointsize=8)
par(mfrow=c(1,1))
low.s = apply(MSPEs.cv, 1, min)
boxplot(MSPEs.cv/low.s, las=2,
        main="RMSPE \n Cross Validation")