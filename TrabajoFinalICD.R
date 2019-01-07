## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

## ------------------------------------------------------------------------
library(scales)
library(dummies)
library(ggplot2)
library(dplyr)
library(tidyr)
library(caret)
library(kknn)
library(MASS)

## ------------------------------------------------------------------------
friedman = read.csv("Datos/friedman/friedman.dat", header = FALSE, comment.char = "@")
head(friedman)

## ------------------------------------------------------------------------
n = length(names(friedman))-1
names(friedman)[1:n] = paste ("X", 1:n, sep="")
names(friedman)[n+1] = "Y"
head(friedman)

## ------------------------------------------------------------------------
dim(friedman)

## ------------------------------------------------------------------------
anyNA(friedman)

## ------------------------------------------------------------------------
summary(friedman)

## ------------------------------------------------------------------------
sapply(friedman, sd)

## ------------------------------------------------------------------------
boxplot(friedman[1:5])
sapply(friedman[1:5], var)

## ------------------------------------------------------------------------
any(duplicated(friedman))

## ------------------------------------------------------------------------
friedman %>% gather() %>% ggplot(aes(value)) +
facet_wrap(~ key, scales = "free") + geom_histogram(bins = 20)

## ------------------------------------------------------------------------
qqnorm(friedman$X1)
qqline(friedman$X1)

qqnorm(friedman$X2)
qqline(friedman$X2)

qqnorm(friedman$X3)
qqline(friedman$X3)

qqnorm(friedman$X4)
qqline(friedman$X4)

qqnorm(friedman$X5)
qqline(friedman$X5)

## ------------------------------------------------------------------------
sapply(friedman[1:5], shapiro.test)

## ------------------------------------------------------------------------
plot(friedman)

## ------------------------------------------------------------------------
plotY = function (x,y) {
  plot(friedman[,y]~friedman[,x], xlab=names(friedman)[x], ylab=names(friedman)[y])
}

par(mfrow=c(2,3))
x = sapply(1:(dim(friedman)[2]-1), plotY, dim(friedman)[2])
par(mfrow=c(1,1))

## ------------------------------------------------------------------------
plotY(4,dim(friedman)[2])

## ------------------------------------------------------------------------
cor(friedman)

## ------------------------------------------------------------------------
australian = read.csv("Datos/australian/australian.dat", header = FALSE, comment.char = "@")
head(australian)

## ------------------------------------------------------------------------
n = length(names(australian))-1
names(australian)[1:n] = paste ("X", 1:n, sep="")
names(australian)[n+1] = "Y"
head(australian)

## ------------------------------------------------------------------------
dim(australian)

## ------------------------------------------------------------------------
anyNA(australian)

## ------------------------------------------------------------------------
summary(australian)

## ------------------------------------------------------------------------
table(australian$Y)

## ------------------------------------------------------------------------
sapply(australian, sd)

## ------------------------------------------------------------------------
boxplot(australian[1:14])
sapply(australian[1:14], var)

## ------------------------------------------------------------------------
australian %>% gather() %>% ggplot(aes(value)) +
facet_wrap(~ key, scales = "free") + geom_histogram(bins = 20)

## ------------------------------------------------------------------------
sapply(australian[c(2,3,5,7,10,13,14)], shapiro.test)

## ------------------------------------------------------------------------
australian$X4 = as.factor(australian$X4)
australian$X12 = as.factor(australian$X12)

australian = dummy.data.frame(australian, sep=".")
head(australian)

## ------------------------------------------------------------------------
maxs = apply(australian, 2, max)
mins = apply(australian, 2, min)
australian = as.data.frame(scale(australian, center = mins, scale = maxs - mins))
head(australian)

## ------------------------------------------------------------------------
correlation = as.data.frame(as.table(cor(australian)))
subset(correlation, abs(Freq) > 0.5 & abs(Freq) < 1.0)

## ------------------------------------------------------------------------
australian = australian[,-which(names(australian) %in% c("X4.1", "X12.1"))]

## ------------------------------------------------------------------------
lmsimple1 = lm(Y~X1, data=friedman)
lmsimple2 = lm(Y~X2, data=friedman)
lmsimple3 = lm(Y~X3, data=friedman)
lmsimple4 = lm(Y~X4, data=friedman)
lmsimple5 = lm(Y~X5, data=friedman)

summary(lmsimple1)
summary(lmsimple2)
summary(lmsimple3)
summary(lmsimple4)
summary(lmsimple5)

## ------------------------------------------------------------------------
lmmultiple1 = lm(Y~., data=friedman)
summary(lmmultiple1)

## ------------------------------------------------------------------------
lmmultiple2 = lm(Y~.-X3, data=friedman)
summary(lmmultiple2)

## ------------------------------------------------------------------------
lmmultiple3 = lm(Y~.-X3-X5, data=friedman)
summary(lmmultiple3)

## ------------------------------------------------------------------------
lminteraccion1 = lm(Y~X4*X1, data=friedman)
summary(lminteraccion1)

## ------------------------------------------------------------------------
lminteraccion2 = lm(Y~X4*X1*X2, data=friedman)
summary(lminteraccion2)

## ------------------------------------------------------------------------
lminteraccion3 = lm(Y~X4*X1*X2*X5, data=friedman)
summary(lminteraccion3)

## ------------------------------------------------------------------------
lmnolinealidad1 = lm(Y~X4+I(X4^2), data=friedman)
summary(lmnolinealidad1)

## ------------------------------------------------------------------------
lmnolinealidad2 = lm(Y~X1+I(X1^2), data=friedman)
summary(lmnolinealidad2)

## ------------------------------------------------------------------------
lmnolinealidad3 = lm(Y~.-X3+I(X1^2), data=friedman)
summary(lmnolinealidad3)

## ------------------------------------------------------------------------
fitknn1 = kknn(Y~., friedman, friedman)

## ------------------------------------------------------------------------
plot(friedman$Y~friedman$X4)
points(friedman$X4, fitknn1$fitted.values, col="red")

## ------------------------------------------------------------------------
RMSE = function(fit, labels) {
  yprime = fit$fitted.values
  sqrt(sum((labels-yprime)^2)/length(yprime)) # RMSE
}

RMSE(fitknn1, friedman$Y)

## ------------------------------------------------------------------------
fitknn2 = kknn(Y~.-X3, friedman, friedman)
RMSE(fitknn2, friedman$Y)

## ------------------------------------------------------------------------
fitknn3 = kknn(Y~.-X3+I(X1^2), friedman, friedman)
RMSE(fitknn3, friedman$Y)

## ------------------------------------------------------------------------
fitknn4 = kknn(Y~.+I(X1^2), friedman, friedman)
RMSE(fitknn4, friedman$Y)

## ------------------------------------------------------------------------
fitknn5 = kknn(Y~.-X2+I(X1^2), friedman, friedman)
fitknn6 = kknn(Y~.-X4+I(X1^2), friedman, friedman)
fitknn7 = kknn(Y~.-X5+I(X1^2), friedman, friedman)
RMSE(fitknn5, friedman$Y)
RMSE(fitknn6, friedman$Y)
RMSE(fitknn7, friedman$Y)

## ------------------------------------------------------------------------
path = "./Datos/friedman/friedman"

run_lm_fold = function(i, x, tt = "test") {
  file = paste(x, "-5-", i, "tra.dat", sep="")
  x_tra = read.csv(file, comment.char="@")
  file = paste(x, "-5-", i, "tst.dat", sep="")
  x_tst = read.csv(file, comment.char="@")
  In = length(names(x_tra)) - 1
  names(x_tra)[1:In] = paste ("X", 1:In, sep="")
  names(x_tra)[In+1] = "Y"
  names(x_tst)[1:In] = paste ("X", 1:In, sep="")
  names(x_tst)[In+1] = "Y"
  
  if (tt == "train") {
    test = x_tra
  }
  else {
    test = x_tst
  }
  
  fitMulti = lm(Y~., x_tra)
  yprime = predict(fitMulti, test)
  sum(abs(test$Y-yprime)^2)/length(yprime) # MSE
}

lmMSEtrain = mean(sapply(1:5, run_lm_fold, path, "train"))
lmMSEtest = mean(sapply(1:5, run_lm_fold, path, "test"))

lmMSEtrain
lmMSEtest

## ------------------------------------------------------------------------
run_knn_fold = function(i, x, tt = "test") {
  file = paste(x, "-5-", i, "tra.dat", sep="")
  x_tra = read.csv(file, comment.char="@")
  file = paste(x, "-5-", i, "tst.dat", sep="")
  x_tst = read.csv(file, comment.char="@")
  In = length(names(x_tra)) - 1
  names(x_tra)[1:In] = paste ("X", 1:In, sep="")
  names(x_tra)[In+1] = "Y"
  names(x_tst)[1:In] = paste ("X", 1:In, sep="")
  names(x_tst)[In+1] = "Y"
  
  if (tt == "train") {
    test = x_tra
  }
  else {
    test = x_tst
  }
  
  fitMulti = kknn(Y~., x_tra, test)
  yprime = fitMulti$fitted.values
  sum(abs(test$Y-yprime)^2)/length(yprime) # MSE
}

knnMSEtrain = mean(sapply(1:5, run_knn_fold, path, "train"))
knnMSEtest = mean(sapply(1:5, run_knn_fold, path, "test"))

knnMSEtrain
knnMSEtest

## ------------------------------------------------------------------------
resultados = read.csv("Datos/regr_train_alumnos.csv")
tablatra = cbind(resultados[,2:dim(resultados)[2]])
colnames(tablatra) = names(resultados)[2:dim(resultados)[2]]
rownames(tablatra) = resultados[,1]

resultados = read.csv("Datos/regr_test_alumnos.csv")
tablatst = cbind(resultados[,2:dim(resultados)[2]])
colnames(tablatst) = names(resultados)[2:dim(resultados)[2]]
rownames(tablatst) = resultados[,1]

## ------------------------------------------------------------------------
difs = (tablatst[,1] - tablatst[,2]) / tablatst[,1]
wilc_1_2 = cbind(ifelse (difs<0, abs(difs)+0.1, 0+0.1),
                 ifelse (difs>0, abs(difs)+0.1, 0+0.1))
colnames(wilc_1_2) = c(colnames(tablatst)[1], colnames(tablatst)[2])
head(wilc_1_2)

## ------------------------------------------------------------------------
LMvsKNNtst = wilcox.test(wilc_1_2[,1], wilc_1_2[,2],
                         alternative = "two.sided", paired=TRUE)
Rmas = LMvsKNNtst$statistic
pvalue = LMvsKNNtst$p.value
KNNvsLMtst = wilcox.test(wilc_1_2[,2], wilc_1_2[,1],
                         alternative = "two.sided", paired=TRUE)
Rmenos = KNNvsLMtst$statistic
Rmas
Rmenos
pvalue

## ------------------------------------------------------------------------
difs = (tablatra[,1] - tablatra[,2]) / tablatra[,1]
wilc_1_2 = cbind(ifelse (difs<0, abs(difs)+0.1, 0+0.1),
                 ifelse (difs>0, abs(difs)+0.1, 0+0.1))
colnames(wilc_1_2) = c(colnames(tablatra)[1], colnames(tablatra)[2])
head(wilc_1_2)

LMvsKNNtst = wilcox.test(wilc_1_2[,1], wilc_1_2[,2],
                         alternative = "two.sided", paired=TRUE)
Rmas = LMvsKNNtst$statistic
pvalue = LMvsKNNtst$p.value
KNNvsLMtst = wilcox.test(wilc_1_2[,2], wilc_1_2[,1],
                         alternative = "two.sided", paired=TRUE)
Rmenos = KNNvsLMtst$statistic
Rmas
Rmenos
pvalue

## ------------------------------------------------------------------------
test_friedman_test = friedman.test(as.matrix(tablatst))
test_friedman_test

## ------------------------------------------------------------------------
test_friedman_train = friedman.test(as.matrix(tablatra))
test_friedman_train

## ------------------------------------------------------------------------
tam = dim(tablatst)
groups = rep(1:tam[2], each=tam[1])
pairwise.wilcox.test(as.matrix(tablatst),
                     groups, p.adjust = "holm", paired = TRUE)

## ------------------------------------------------------------------------
tam = dim(tablatra)
groups = rep(1:tam[2], each=tam[1])
pairwise.wilcox.test(as.matrix(tablatra),
                     groups, p.adjust = "holm", paired = TRUE)

## ------------------------------------------------------------------------
set.seed(1010)
shuffle_ds = sample(dim(australian)[1]) 
eightypct = (dim(australian)[1] * 80) %/% 100
australian[,dim(australian)[2]] = as.factor(australian[,dim(australian)[2]])


## ------------------------------------------------------------------------
head(australian)

## ------------------------------------------------------------------------
aust_train = australian[shuffle_ds[1:eightypct], 1:dim(australian)[2]-1]
aust_test = australian[shuffle_ds[(eightypct+1):dim(australian)[1]],1:dim(australian)[2]-1] 

aust_train_labels = australian[shuffle_ds[1:eightypct], dim(australian)[2]]
aust_test_labels = australian[shuffle_ds[(eightypct+1):dim(australian)[1]],dim(australian)[2]]

## ------------------------------------------------------------------------
knnModel = train(aust_train, aust_train_labels, method="knn",
           metric="Accuracy", tuneGrid = data.frame(.k=seq(1,15,2))) 
knnModel

## ------------------------------------------------------------------------
mejoresK = knnModel$results$k[order(knnModel$results$Accuracy, decreasing = TRUE)[1:3]]
mejoresK

knnModel.1 = train(aust_train, aust_train_labels, method="knn",
               metric="Accuracy", tuneGrid = data.frame(.k=mejoresK[1]))

knnModel.2 = train(aust_train, aust_train_labels, method="knn",
               metric="Accuracy", tuneGrid = data.frame(.k=mejoresK[2]))

knnModel.3 = train(aust_train, aust_train_labels, method="knn",
               metric="Accuracy", tuneGrid = data.frame(.k=mejoresK[3]))

## ------------------------------------------------------------------------
knnPred.1 = predict(knnModel.1, aust_test)
knnPred.2 = predict(knnModel.2, aust_test)
knnPred.3 = predict(knnModel.3, aust_test)

## ------------------------------------------------------------------------
postResample(knnPred.1, aust_test_labels)
postResample(knnPred.2, aust_test_labels)
postResample(knnPred.3, aust_test_labels)

## ------------------------------------------------------------------------
par(mfrow=c(2,2))
plot(aust_test$X13~aust_test$X2,col=knnPred.1, main="Modelo 11-NN")
plot(aust_test$X13~aust_test$X2,col=knnPred.2, main="Modelo 13-NN")
plot(aust_test$X13~aust_test$X2,col=knnPred.3, main="Modelo 9-NN")
plot(aust_test$X13~aust_test$X2,col=aust_test_labels, main="Originales Test")
par(mfrow=c(1,1))

## ------------------------------------------------------------------------
knnPredTra.1 = predict(knnModel.1, aust_train)
knnPredTra.2 = predict(knnModel.2, aust_train)
knnPredTra.3 = predict(knnModel.3, aust_train)

postResample(knnPredTra.1, aust_train_labels)
postResample(knnPredTra.2, aust_train_labels)
postResample(knnPredTra.3, aust_train_labels)

## ------------------------------------------------------------------------
sapply(australian[1:dim(australian)[2]-1], var)

## ------------------------------------------------------------------------
Y = aust_train_labels
lda.fit = lda(Y~., data=cbind(aust_train,Y))
lda.fit

## ------------------------------------------------------------------------
plot(lda.fit, type="both")

## ------------------------------------------------------------------------
lda.pred = predict(lda.fit, aust_test)

table(lda.pred$class,aust_test_labels)
ldaPred = mean(lda.pred$class==aust_test_labels)
ldaPred

## ------------------------------------------------------------------------
lda.predTra = predict(lda.fit, aust_train)

table(lda.predTra$class,aust_train_labels)
ldaPredTra = mean(lda.predTra$class==aust_train_labels)
ldaPredTra

## ------------------------------------------------------------------------
aust_0 = australian[australian$Y == 0,][1:dim(australian)[2]-1]
aust_1 = australian[australian$Y == 1,][1:dim(australian)[2]-1]

sapply(aust_0, var)
sapply(aust_1, var)

## ------------------------------------------------------------------------
qda.fit = qda(Y~.-X4.3, data=cbind(aust_train,Y))
qda.fit

## ------------------------------------------------------------------------
qda.pred = predict(qda.fit, aust_test)

table(qda.pred$class, aust_test_labels)
qdaPred = mean(qda.pred$class==aust_test_labels)
qdaPred

## ------------------------------------------------------------------------
qda.predTra = predict(qda.fit, aust_train)

table(qda.predTra$class,aust_train_labels)
qdaPredTra = mean(qda.predTra$class==aust_train_labels)
qdaPredTra

## ------------------------------------------------------------------------
resultados = read.csv("Datos/clasif_train_alumnos.csv")
tablatra = cbind(resultados[,2:dim(resultados)[2]])
colnames(tablatra) = names(resultados)[2:dim(resultados)[2]]
rownames(tablatra) = resultados[,1]

resultados = read.csv("Datos/clasif_test_alumnos.csv")
tablatst = cbind(resultados[,2:dim(resultados)[2]])
colnames(tablatst) = names(resultados)[2:dim(resultados)[2]]
rownames(tablatst) = resultados[,1]

## ------------------------------------------------------------------------
difs = (tablatra[,1] - tablatra[,2]) / tablatra[,1]
wilc_1_2 = cbind(ifelse (difs<0, abs(difs)+0.1, 0+0.1),
                 ifelse (difs>0, abs(difs)+0.1, 0+0.1))
colnames(wilc_1_2) = c(colnames(tablatra)[1], colnames(tablatra)[2])
head(wilc_1_2)

KNNvsLDAtst = wilcox.test(wilc_1_2[,1], wilc_1_2[,2],
                         alternative = "two.sided", paired=TRUE)
Rmas = KNNvsLDAtst$statistic
pvalue = KNNvsLDAtst$p.value
LDAvsKNNtst = wilcox.test(wilc_1_2[,2], wilc_1_2[,1],
                         alternative = "two.sided", paired=TRUE)
Rmenos = LDAvsKNNtst$statistic
Rmas
Rmenos
pvalue

## ------------------------------------------------------------------------
difs = (tablatst[,1] - tablatst[,2]) / tablatst[,1]
wilc_1_2 = cbind(ifelse (difs<0, abs(difs)+0.1, 0+0.1),
                 ifelse (difs>0, abs(difs)+0.1, 0+0.1))
colnames(wilc_1_2) = c(colnames(tablatst)[1], colnames(tablatst)[2])
head(wilc_1_2)

KNNvsLDAtst = wilcox.test(wilc_1_2[,1], wilc_1_2[,2],
                         alternative = "two.sided", paired=TRUE)
Rmas = KNNvsLDAtst$statistic
pvalue = KNNvsLDAtst$p.value
LDAvsKNNtst = wilcox.test(wilc_1_2[,2], wilc_1_2[,1],
                         alternative = "two.sided", paired=TRUE)
Rmenos = LDAvsKNNtst$statistic
Rmas
Rmenos
pvalue

## ------------------------------------------------------------------------
difs = (tablatra[,1] - tablatra[,3]) / tablatra[,1]
wilc_1_2 = cbind(ifelse (difs<0, abs(difs)+0.1, 0+0.1),
                 ifelse (difs>0, abs(difs)+0.1, 0+0.1))
colnames(wilc_1_2) = c(colnames(tablatra)[1], colnames(tablatra)[3])
head(wilc_1_2)

KNNvsQDAtst = wilcox.test(wilc_1_2[,1], wilc_1_2[,2],
                         alternative = "two.sided", paired=TRUE)
Rmas = KNNvsQDAtst$statistic
pvalue = KNNvsQDAtst$p.value
QDAvsKNNtst = wilcox.test(wilc_1_2[,2], wilc_1_2[,1],
                         alternative = "two.sided", paired=TRUE)
Rmenos = QDAvsKNNtst$statistic
Rmas
Rmenos
pvalue

## ------------------------------------------------------------------------
difs = (tablatst[,1] - tablatst[,3]) / tablatst[,1]
wilc_1_2 = cbind(ifelse (difs<0, abs(difs)+0.1, 0+0.1),
                 ifelse (difs>0, abs(difs)+0.1, 0+0.1))
colnames(wilc_1_2) = c(colnames(tablatst)[1], colnames(tablatst)[3])
head(wilc_1_2)

KNNvsQDAtst = wilcox.test(wilc_1_2[,1], wilc_1_2[,2],
                         alternative = "two.sided", paired=TRUE)
Rmas = KNNvsQDAtst$statistic
pvalue = KNNvsQDAtst$p.value
QDAvsKNNtst = wilcox.test(wilc_1_2[,2], wilc_1_2[,1],
                         alternative = "two.sided", paired=TRUE)
Rmenos = QDAvsKNNtst$statistic
Rmas
Rmenos
pvalue

## ------------------------------------------------------------------------
difs = (tablatra[,2] - tablatra[,3]) / tablatra[,2]
wilc_1_2 = cbind(ifelse (difs<0, abs(difs)+0.1, 0+0.1),
                 ifelse (difs>0, abs(difs)+0.1, 0+0.1))
colnames(wilc_1_2) = c(colnames(tablatra)[2], colnames(tablatra)[3])
head(wilc_1_2)

LDAvsQDAtst = wilcox.test(wilc_1_2[,1], wilc_1_2[,2],
                         alternative = "two.sided", paired=TRUE)
Rmas = LDAvsQDAtst$statistic
pvalue = LDAvsQDAtst$p.value
QDAvsLDAtst = wilcox.test(wilc_1_2[,2], wilc_1_2[,1],
                         alternative = "two.sided", paired=TRUE)
Rmenos = QDAvsLDAtst$statistic
Rmas
Rmenos
pvalue

## ------------------------------------------------------------------------
difs = (tablatst[,2] - tablatst[,3]) / tablatst[,2]
wilc_1_2 = cbind(ifelse (difs<0, abs(difs)+0.1, 0+0.1),
                 ifelse (difs>0, abs(difs)+0.1, 0+0.1))
colnames(wilc_1_2) = c(colnames(tablatst)[2], colnames(tablatst)[3])
head(wilc_1_2)

LDAvsQDAtst = wilcox.test(wilc_1_2[,1], wilc_1_2[,2],
                         alternative = "two.sided", paired=TRUE)
Rmas = LDAvsQDAtst$statistic
pvalue = LDAvsQDAtst$p.value
QDAvsLDAtst = wilcox.test(wilc_1_2[,2], wilc_1_2[,1],
                         alternative = "two.sided", paired=TRUE)
Rmenos = QDAvsLDAtst$statistic
Rmas
Rmenos
pvalue

## ------------------------------------------------------------------------
test_friedman_train = friedman.test(as.matrix(tablatra))
test_friedman_train

## ------------------------------------------------------------------------
test_friedman_test = friedman.test(as.matrix(tablatst))
test_friedman_test

## ------------------------------------------------------------------------
tam = dim(tablatra)
groups = rep(1:tam[2], each=tam[1])
pairwise.wilcox.test(as.matrix(tablatra),
                     groups, p.adjust = "holm", paired = TRUE)

## ------------------------------------------------------------------------
tam = dim(tablatst)
groups = rep(1:tam[2], each=tam[1])
pairwise.wilcox.test(as.matrix(tablatst),
                     groups, p.adjust = "holm", paired = TRUE)

