import prediction_function as pred

#number param as which layer to start training
#predict from dataset without augmentation
for i in  range(26,33,2):
    pred.predictCustlr(paramstart=i)
    