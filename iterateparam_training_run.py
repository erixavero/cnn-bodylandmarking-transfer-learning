import transfertraining_function as tff

#number param as which layer to start training
numberoftraining = 0
for i in  range(26,33,2):
    tff.starttraining(trainstartparam=i,experiment='iterateparam_models/')
    numberoftraining=numberoftraining+1
    
print('total experiment iterated: ',numberoftraining)