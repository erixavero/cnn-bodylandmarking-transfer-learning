import torch
import time
import pandas as pd
from modelclass import LandmarkExpNetwork as LMNET
from custlrloader import CustlrDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import src.const as const
from src.utils import LandmarkEvaluator as _evaluator

def starttraining(trainstartparam=26,traincsv='dataset/custlr/auged/train.csv',valcsv='dataset/custlr/auged/val.csv', base_path = 'dataset/custlr/auged/', experiment=''):
    evaluator = _evaluator()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #load empty model
    model = LMNET()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    statedict = torch.load('models/lmfulltrain.pkl')
    
    #load model state
    model.load_state_dict(statedict)
    
    print(model)
    #freeze some layer params total 0-51 vgg16 extractor 0-25, lm branch upsample 26-51
    for i, param in enumerate(model.parameters()):
        if i >= trainstartparam:
            break
        param.requires_grad = False
    
    k=trainstartparam
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(k,name)
            k=k+1
        
    #print(model.lm_branch.conv10.weight)
    
    model.to(device)
    
    #load dataset
    traindf = pd.read_csv(traincsv)
    valdf = pd.read_csv(valcsv)
    
    train_dataset = CustlrDataset(traindf, mode='RANDOM', base_path=base_path)
    val_dataset = CustlrDataset(valdf, mode='LARGESTCENTER', base_path=base_path)
    
    print('training set', len(train_dataset))
    print('val set', len(val_dataset))
    dssize = len(train_dataset)+len(val_dataset)
    
    trainloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    valloader = DataLoader(val_dataset, batch_size=10, shuffle=True, num_workers=4)
        
    logname = time.strftime('%m-%d %H:%M:%S', time.localtime())
    logname = logname + ' param ' + str(trainstartparam)+'-51' + 'dataset size '+ str(dssize)
    logname = 'runs/'+logname
    writer = SummaryWriter(logname)
    
    #retrain last layer
    train_pos_loss = []
    val_pos_loss = []
    NUMEPOCH = 20
    NUMSTEP = len(trainloader)
    step = 0
    for epoch in range(NUMEPOCH):
        for stat in ['train','val']:
            print(stat)
            if stat == 'train':
                model.train()
                for i, sample in enumerate(trainloader):
                    step += 1
                    for key in sample:
                        sample[key] = sample[key].to(device)
                    output = model(sample)
                    #print(output['lm_pos_map'])
                    
                    loss = model.cal_loss(sample, output)
                    
                    optimizer.zero_grad()
                    loss['all'].backward()
                    optimizer.step()
                    
                    if 'lm_vis_loss' in loss:
                        writer.add_scalar('loss/lm_vis_loss', loss['lm_vis_loss'], step)
                        writer.add_scalar('loss_weighted/lm_vis_loss', loss['weighted_lm_vis_loss'], step)
                    if 'lm_pos_loss' in loss:
                        writer.add_scalar('loss/lm_pos_loss', loss['lm_pos_loss'], step)
                        writer.add_scalar('loss_weighted/lm_pos_loss', loss['weighted_lm_pos_loss'], step)
                    writer.add_scalar('loss_weighted/all', loss['all'], step)
                    writer.add_scalar('global/learning_rate', learning_rate, step)
                    
                    train_pos_loss.append(loss['lm_pos_loss'])
                    print('EPOCH:', epoch+1,'/',NUMEPOCH,'step:', i+1,'/',NUMSTEP,'| lm position loss:', loss['lm_pos_loss'],' | weighted loss:', loss['weighted_lm_pos_loss'])
            else:
                model.eval()
                for i, sample in enumerate(valloader):
                    step += 1
                    for key in sample:
                        sample[key] = sample[key].to(device)
                    output = model(sample)
                    #print(output['lm_pos_map'])
                    evaluator.add(output, sample)
                ret = evaluator.evaluate()
                for i in range(8):
                    print('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i])
                    writer.add_scalar('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i], step)
                print('metrics/dist_all', ret['lm_dist'])
                writer.add_scalar('metrics/dist_all', ret['lm_dist'], step)
                val_pos_loss.append(ret['lm_dist'])
                print('EPOCH:', epoch+1,'/',NUMEPOCH,'| lm distance:', ret['lm_dist'])
                model.train()
        learning_rate *= const.LEARNING_RATE_DECAY
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    torch.save(model.state_dict(), 'models/'+ experiment + 'lm param ' + str(trainstartparam) + '-51'+' dataset '+str(dssize) +'.pkl')
    plt.plot(train_pos_loss,label='training pos loss')
    #tfigname = 'experimentlayersgraph/training loss'+str(trainstartparam)+'-51'+' dataset '+str(dssize) +'.png'
    #plt.savefig(tfigname)
    plt.show()
    plt.plot(val_pos_loss,label='validation pos loss')
    #vfigname = 'experimentlayersgraph/val loss'+str(trainstartparam)+'-51'+' dataset '+str(dssize) +'.png'
    #plt.savefig(vfigname)
    plt.show()
