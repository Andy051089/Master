library(caret)
cvd=read.csv("D:/研究所/1131/統計分析與軟體實作/講義/CVD_All.csv",header=T)
cvd=cvd[complete.cases(cvd), ] 
head(cvd)
names(cvd)
table(cvd$smoking)
cvd$smoking_level=as.factor(cvd$smoking_level)

set.seed(42)
train.index <- sample(x=1:nrow(cvd), size=ceiling(0.8*nrow(cvd) ))
cvd_train = cvd[train.index, ]
cvd_test = cvd[-train.index, ]

fullmod = glm(CVD ~ .-smoking_level-ID ,family=binomial, data=cvd_train)
summary(fullmod)

backwards = step(fullmod,trace=0)
formula(backwards)
summary(backwards)
backwards$deviance
#大模型為使用所有特徵
#小模型為backwise選擇的特徵
anova(fullmod,backwards, test="Chisq")
#P>0.05無法拒絕虛無假設
#使用小模型
cvd_test$predict_b=predict(backwards, newdata=cvd_test, type="response")
(tab_b=table(cvd_test$CVD, 0+(cvd_test$predict_b>0.15)))
hist(cvd_test$predict_b)

pred=as.factor(0+(cvd_test$predict_b>0.15))  
truth=as.factor(cvd_test$CVD)
length(truth);length(pred)
index=confusionMatrix(pred, truth, positive = "1")
index
F1=(2*index$byClass[5]*index$byClass[1])/(index$byClass[5]+index$byClass[1])
F1