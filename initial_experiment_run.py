import transfertraining_function as tff
import prediction_function as pred

#run training function
tff.starttraining(trainstartparam=50,traincsv='dataset/custlr/auged/custlrinfo101.csv',valcsv='dataset/custlr/auged/custlrinfo101.csv')

pred.predictCustlr(paramstart=50, predcsv='dataset/custlr/auged/custlrinfo101.csv',dssize=198,experiment='')