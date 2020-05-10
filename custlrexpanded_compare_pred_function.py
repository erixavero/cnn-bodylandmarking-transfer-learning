import torch
import pandas as pd
import numpy as np
from modelclass import LandmarkExpNetwork as LMNET
from custlrloader import CustlrDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from src.utils import LandmarkEvaluator as _evaluator

#26 start of lmbranch
def comparePrediction(paramstart=26, predcsv='dataset/custlr/auged/test.csv'):
    evaluator = _evaluator()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #load empty model
    model1 = LMNET()
    model2 = LMNET()
    
    dictname1 = 'models/custlrexpanded_models/lm param 26-51 dataset 11060.pkl'
    dictname2 = 'models/custlrexpanded_models/lm param 26-51 dataset 22120.pkl'
    statedict1 = torch.load(dictname1)
    statedict2 = torch.load(dictname2)
    
    val_pos_loss = []
    #load model state
    model1.load_state_dict(statedict1)
    model2.load_state_dict(statedict2)
        
    df = pd.read_csv(predcsv)
    df = df.iloc[:3,:]
    dataset = CustlrDataset(df, mode='LARGESTCENTER')
    print('dataset', len(dataset))
    
    for i in range(len(dataset)):
        dataset.plot_landmark_map(i)
        predname = 'pred_visualized/custlrexpanded_plot/'+'lmap gt ' + 'img'+str(i)+'.jpg'
        plt.savefig(predname)
    
    #dataset.plot_landmark_map(0)
    #dataset.plot_sample(0)
    ldr = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    EPOCH = 1
    val_pos_loss = []
    
    model1.eval()
    model1.to(device)
    model2.eval()
    model2.to(device)
    
    for e in range(EPOCH):
        for i, sample in enumerate(ldr):
            for key in sample:
                sample[key] = sample[key].to(device)
            output1 = model1(sample)
            output2 = model2(sample)
            
            #print(output['lm_pos_output'][0])
            dataset.plot_sample(i)
            plt.scatter(output1['lm_pos_output'][0,:,0]*224,output1['lm_pos_output'][0,:,1]*224, s=5, color = 'blue')
            plt.scatter(output2['lm_pos_output'][0,:,0]*224,output2['lm_pos_output'][0,:,1]*224, s=5, color = 'red')
            predname = 'pred_visualized/custlrexpanded_plot/'+'param ' + str(paramstart)+'-51' + 'img'+str(i)+'.jpg'
            plt.savefig(predname)
            plt.show()
            
            l_map1 = np.max(output1['lm_pos_map'][0].cpu().detach().numpy(), axis=0)
            l_map2 = np.max(output2['lm_pos_map'][0].cpu().detach().numpy(), axis=0)
            plt.imshow(l_map1)
            predname = 'pred_visualized/custlrexpanded_plot/'+'lmap reg ' + 'img'+str(i)+'.jpg'
            #plt.savefig(predname)
            plt.show()
            plt.imshow(l_map2)
            predname = 'pred_visualized/custlrexpanded_plot/'+'lmap exp ' + 'img'+str(i)+'.jpg'
            #plt.savefig(predname)
            plt.show()
            
            evaluator.add(output2, sample)    
        ret = evaluator.evaluate()
        val_pos_loss.append(ret['lm_dist'])
        print(ret['lm_dist'])
    
    avgloss = sum(val_pos_loss)/len(val_pos_loss)
    print('avg: ',avgloss)
    
    