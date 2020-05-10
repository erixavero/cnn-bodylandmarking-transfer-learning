import torch
import pandas as pd
import numpy as np
from modelclass import LandmarkExpNetwork as LMNET
from custlrloader import CustlrDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from src.utils import LandmarkEvaluator as _evaluator

#26 start of lmbranch
def predictCustlr(paramstart=26, predcsv='dataset/custlr/auged/test.csv', dssize=11060, experiment='iterateparam_models/'):
    evaluator = _evaluator()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #load empty model
    model = LMNET()
    
    dictname = 'models/'+ experiment + 'lm param ' + str(paramstart) + '-51'+' dataset '+str(dssize) +'.pkl'
    statedict = torch.load(dictname)
    
    val_pos_loss = []
    #load model state
    model.load_state_dict(statedict)
    for i, param in enumerate(model.parameters()):
        param.requires_grad = False
        
    df = pd.read_csv(predcsv)
    df = df.iloc[:3,:]
    dataset = CustlrDataset(df, mode='LARGESTCENTER')
    print('dataset', len(dataset))
    
    #dataset.plot_landmark_map(0)
    #dataset.plot_sample(0)
    ldr = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    EPOCH = 1
    val_pos_loss = []
    
    model.eval()
    model.to(device)
    
    for e in range(EPOCH):
        for i, sample in enumerate(ldr):
            for key in sample:
                sample[key] = sample[key].to(device)
            output = model(sample)
            
            print(output['lm_pos_output'][0])
            dataset.plot_sample(i)
            plt.scatter(output['lm_pos_output'][0,:,0]*224,output['lm_pos_output'][0,:,1]*224, s=5)
            predname = 'pred_visualized/iterateparam_plot/'+'param ' + str(paramstart)+'-51' + 'img'+str(i)+'.jpg'
            plt.savefig(predname)
            plt.show()
            
    landmark_map = np.max(output['lm_pos_map'][0].cpu().numpy(), axis=0)
    #plt.imshow(landmark_map)
    