import transfertraining_function as tff
import custlrexpanded_compare_pred_function as cop

#training with default and expanded dataset
tff.starttraining(traincsv='dataset/custlr/auged/train.csv',valcsv='dataset/custlr/auged/val.csv',experiment='custlrexpanded_models/')
tff.starttraining(traincsv='dataset/custlr/auged/train2x.csv',valcsv='dataset/custlr/auged/val2x.csv',experiment='custlrexpanded_models/')

cop.comparePrediction(predcsv='dataset/custlr/auged/test.csv')