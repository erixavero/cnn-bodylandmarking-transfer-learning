import prediction_function as pred

#number param as which layer to start training
#predict from dataset without augmentation

testcsv = 'dataset/custlr/auged/test13lm.csv'
traincsv = 'dataset/custlr/auged/train13lm.csv'

print(testcsv)
pred.predictCustlr(predcsv=testcsv, dssize=22120, experiment='custlrexpanded_models/')

print(traincsv)
pred.predictCustlr(predcsv=traincsv, dssize=22120, experiment='custlrexpanded_models/')
 