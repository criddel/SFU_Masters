# Some code adapted from STAT 652 Fall 2020 - T. Loughin
## Hyperparameter Tuning for Gradient Boost Model

data = read.csv('fish.csv')
library(caret)
data$Species = as.factor(data$Species)
dmy = dummyVars("~.", data=data)
data = data.frame(predict(dmy, newdata = data))
#data = data[,-1]

V=5
R=2 
n2 = nrow(data)
# Create the folds and save in a matrix
folds = matrix(NA, nrow=n2, ncol=R)
for(r in 1:R){
  folds[,r]=floor((sample.int(n2)-1)*V/n2) + 1
}

shr = c(.001,.005,.025)
dep = c(5, 10,25,40,50)
### Second grid 
#dep = c(1,2,3,4)
#shr = c(0.0025, 0.005, 0.0075, 0.01)
trees = 10000

NS = length(shr)
ND = length(dep)
gb.cv = matrix(NA, nrow=ND*NS, ncol=V*R)
opt.tree = matrix(NA, nrow=ND*NS, ncol=V*R)

qq = 1
for(r in 1:R){
  for(v in 1:V){
    pro.train = data[folds[,r]!=v,]
    pro.test = data[folds[,r]==v,]
    counter=1
    for(d in dep){
      for(s in shr){
        pro.gbm <- gbm(data=pro.train, log_rarefied_abundance~., distribution="gaussian", 
                       n.trees=trees, interaction.depth=d, shrinkage=s, 
                       bag.fraction=0.8)
        treenum = min(trees, 2*gbm.perf(pro.gbm, method="OOB", plot.it=FALSE))
        opt.tree[counter,qq] = treenum
        preds = predict(pro.gbm, newdata=pro.test, n.trees=treenum)
        gb.cv[counter,qq] = mean((preds - pro.test$log_rarefied_abundance)^2)
        counter=counter+1
      }
    }
    qq = qq+1
  }  
}

parms = expand.grid(shr,dep)
row.names(gb.cv) = paste(parms[,2], parms[,1], sep="|")
row.names(opt.tree) = paste(parms[,2], parms[,1], sep="|")

opt.tree
gb.cv

(mean.tree = apply(opt.tree, 1, mean))
(mean.cv = sqrt(apply(gb.cv, 1, mean)))
min.cv = apply(gb.cv, 2, min)

x11(h=7,w=10,pointsize=8)
boxplot(sqrt(gb.cv), use.cols=FALSE, las=2)

x11(h=7,w=10,pointsize=8)
boxplot(sqrt(t(gb.cv)), use.cols=TRUE, las=2, 
        main="GBM Fine-Tuning Variables and Node Sizes")
