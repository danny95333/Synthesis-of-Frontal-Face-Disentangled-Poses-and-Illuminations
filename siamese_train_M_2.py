#from __future__ import print_function
import cv2
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchnet.meter as meter
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from dataset import multiPIE
from siamese_model_2nd import Siamese
from contrastive import ContrastiveLoss
import numpy as np
# import cv2
#from pycrayon import CrayonClient
 
#for plotting loss
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time,math
from logger import Logger
# from models_Parsing import ParseNet
saveFile = open('/home/shumao/wyw_files/siamese_output_M_2/record.txt', 'w')
saveFile.write("niter:" + str(50000) + "\n")
saveFile.write("---lr:" + str(0.0001) + "\n")
saveFile.write("beta1:" + str(0.7) + "\n")
saveFile.write("W:-1-x-x-x-x-x-" + "\n")
saveFile.write("L2 loss" + "\n")
saveFile.write("after load model from: train-3-28000pth")
logger = Logger('./log_1');

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--loadSize', type=int, default=100, help='the height / width of the input image to network')
parser.add_argument('--fineSize', type=int, default=96, help='the height / width of the input image to network')
parser.add_argument('--id_num', type=int, default=200, help='Total training identity.')
parser.add_argument('--pose_num', type=int, default=9, help='Total training pose.')
parser.add_argument('--light_num', type=int, default=20, help='Total training light.')
parser.add_argument('--niter', type=int, default=50000, help='number of iterations to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.7, help='beta1 for adam. default=0.7')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='/home/shumao/wyw_files/siamese_output_M_2', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataPath', default='/home/shumao/dr-gan/Data_new_realigned2/setting2/train/', help='which dataset to train on')
parser.add_argument('--modelPath', default='/home/shumao/wyw_files/siamese_output_3/netS_28000.pth', help='which dataset to train on')
parser.add_argument('--save_step', type=int, default=400, help='save weights every 400 iterations ')
parser.add_argument('--labelPath', default='/home/shumao/dr-gan/Data_new_realigned2/setting2/Facedata/', help='which dataset to train on')


opt = parser.parse_args()
print(opt) # print every parser arguments
# print(opt.niter)


try:
    os.makedirs(opt.outf)
except OSError:
    pass

w_r = 1
# w_cL = 0.02
# w_cP = 0.02
# w_cI = 0.02
# w_P = 0.02
# w_L = 0.02

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
#---------------------Load Mask-------------------
mask = np.load('mask_20.npy')
mask = mask.astype(np.float32)
M = torch.from_numpy(mask.transpose((2, 0, 1)))
FinalMask = M.expand(opt.batchSize,3,96,96)
# print m.size()
# 3x96x96


#---------------------Load DATA-------------------------
dataset_1 = multiPIE(opt.dataPath,opt.loadSize,opt.fineSize,labelPath = opt.labelPath)
# dataset_2 = multiPIE(opt.dataPath,opt.loadSize,opt.fineSize,opt.labelPath)
dataset_test = multiPIE('/home/shumao/dr-gan/comparison/',opt.loadSize,opt.fineSize,labelPath = opt.labelPath)
loader_train_1 = torch.utils.data.DataLoader(dataset=dataset_1,
                                      batch_size = opt.batchSize,
                                      shuffle=True,
                                      num_workers=4,
                                      drop_last = True)
# loader_train_2 = torch.utils.data.Dataloader(dataset=dataset_1,
#                                       batch_size = opt.batchSize,
#                                       shuffle=True,
#                                       num_workers=4)


loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                          batch_size = 9,
                                          shuffle=False,
                                          num_workers=4)
data_train_1 = iter(loader_train_1)
# data_train_2 = iter(loader_train_2)
data_test = iter(loader_test)


#----------------------Parameters-----------------------
num_pose = opt.pose_num
num_light = opt.light_num
num_iden = opt.id_num


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') !=-1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') !=-1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)




netS = Siamese()
netS.load_state_dict(torch.load(opt.modelPath))

#-----------------params freeze-----------------
for param in netS.conv11.parameters():
    param.requires_grad = False
for param in netS.conv1r.parameters():
    param.requires_grad = False
for param in netS.conv12.parameters():
    param.requires_grad = False
for param in netS.conv21.parameters():
    param.requires_grad = False
for param in netS.conv22.parameters():
    param.requires_grad = False
for param in netS.conv23.parameters():
    param.requires_grad = False
for param in netS.conv31.parameters():
    param.requires_grad = False
for param in netS.conv32.parameters():
    param.requires_grad = False
for param in netS.conv33.parameters():
    param.requires_grad = False
for param in netS.conv41.parameters():
    param.requires_grad = False
for param in netS.conv42.parameters():
    param.requires_grad = False
for param in netS.conv43.parameters():
    param.requires_grad = False
for param in netS.conv51.parameters():
    param.requires_grad = False
for param in netS.conv52.parameters():
    param.requires_grad = False
for param in netS.conv53.parameters():
    param.requires_grad = False
for param in netS.convfc.parameters():
    param.requires_grad = False


#-----------------params freeze-----------------
if(opt.cuda):
    netS.cuda()

#-------------------Loss & Optimization

optimizerS = torch.optim.Adam(filter(lambda p: p.requires_grad, netS.parameters()),lr=opt.lr, betas=(opt.beta1, 0.999))

poss_contrastive_loss = ContrastiveLoss() # load from the begining
light_contrastive_loss = ContrastiveLoss()
identity_contrastive_loss = ContrastiveLoss()
reconstructe_loss = nn.MSELoss()
pose_class_loss = nn.CrossEntropyLoss()
light_class_loss = nn.CrossEntropyLoss()

#------------------ Global Variables------------------
input_pose_1 = torch.LongTensor(opt.batchSize)
input_light_1 = torch.LongTensor(opt.batchSize)
# input_pose_2 = torch.LongTensor(opt.batchSize)
# input_light_2 = torch.LongTensor(opt.batchSize)

inputImg_1 = torch.FloatTensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
inputImg_2 = torch.FloatTensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
GT = torch.FloatTensor(opt.batchSize, 3,opt.fineSize, opt.fineSize)
same_pose = torch.FloatTensor(opt.batchSize)
same_iden = torch.FloatTensor(opt.batchSize)
same_light = torch.FloatTensor(opt.batchSize)

# w_1 = torch.FloatTensor(1)
# w_2 = torch.FloatTensor(20)
# w_3 = torch.FloatTensor(10)
# w_4 = torch.FloatTensor(10)
# w_5 = torch.FloatTensor(10)
# w_6 = torch.FloatTensor(20)
# output_pose_1_label = torch.LongTensor(opt.batchSize)
# output_pose_2_label = torch.LongTensor(opt.batchSize)
# output_light_1_label = torch.LongTensor(opt.batchSize)
# output_light_2_label = torch.LongTensor(opt.batchSize)

input_pose_1 = Variable(input_pose_1)
# input_pose_2 = Variable(input_pose_2)
input_light_1 = Variable(input_light_1)
# input_light_2 = Variable(input_light_2)

inputImg_1 = Variable(inputImg_1)
inputImg_2 = Variable(inputImg_2)
GT = Variable(GT)
same_pose = Variable(same_pose)
same_iden = Variable(same_iden)
same_light = Variable(same_light)

FinalMask = Variable(FinalMask)

# w_1 = Variable(w_1, requires_grad = False)
# w_2 = Variable(w_2, requires_grad = False)
# w_3 = Variable(w_3, requires_grad = False)
# w_4 = Variable(w_4, requires_grad = False)
# w_5 = Variable(w_5, requires_grad = False)
# w_6 = Variable(w_6, requires_grad = False)


pose_mtr = meter.ConfusionMeter(k=opt.pose_num)
light_mtr = meter.ConfusionMeter(k=opt.light_num)


if(opt.cuda):

    input_pose_1 = input_pose_1.cuda()
    # input_pose_2 = input_pose_2.cuda()
    input_light_1 = input_light_1.cuda()
    # input_light_2 = input_light_2.cuda()
    inputImg_1 = inputImg_1.cuda()
    inputImg_2 = inputImg_2.cuda()
    GT = GT.cuda()
    same_pose = same_pose.cuda()
    same_light = same_light.cuda()
    same_iden = same_iden.cuda()

    FinalMask = FinalMask.cuda()

    # w_1 = w_1.cuda()
    # w_2 = w_1.cuda()
    # w_3 = w_1.cuda()
    # w_4 = w_1.cuda()
    # w_5 = w_1.cuda()
    # w_6 = w_1.cuda()
    # poss_contrastive_loss.cuda()
    # light_contrastive_loss.cuda()
    # identity_contrastive_loss.cuda()
    pose_class_loss.cuda()
    light_class_loss.cuda()
    reconstructe_loss.cuda()


#------------------test---------

# k = 0 # for meter

err_total = 0
err_recon = 0
err_contraL = 0
err_contraP = 0
err_contraI = 0
err_classP = 0
err_classL = 0

def test(iteration, data_test, loader_test):
    try:
        images_1,po_1,li_1,GT_1,by_image,same_po,same_li,same_id = data_test.next()
    except StopIteration:
        data_test = iter(loader_test)
        images_1,po_1,li_1,GT_1,by_image,same_po,same_li,same_id = data_test.next()

    GT.data.resize_(GT_1.size()).copy_(GT_1)
    inputImg_1.data.resize_(images_1.size()).copy_(images_1)
    inputImg_2.data.resize_(by_image.size()).copy_(by_image)
    input_pose_1.data.resize_(po_1.size()).copy_(po_1)
    input_light_1.data.resize_(li_1.size()).copy_(li_1)


    output_pose_1, output_pose_2, output_light_1, output_light_2, out_f_1, out_f_2, out = netS(inputImg_1, inputImg_2)
    vutils.save_image(out.data,
        '%s/fake_samples_iteration_%03d.png' % (opt.outf, iteration), normalize=True)
    vutils.save_image(inputImg_1.data,
            '%s/input_samples_iteration_%03d.png' % (opt.outf, iteration), normalize=True)



#-------------------train----------------------
for iteration in range(1,opt.niter+1):
    running_corrects = 0
    running_corrects_light = 0
    try:
        images_1,po_1,li_1,GT_1,by_image,same_po,same_li,same_id= data_train_1.next()
    except StopIteration:
        data_train_1 = iter(loader_train_1)
        images_1,po_1,li_1,GT_1,by_image,same_po,same_li,same_id = data_train_1.next()

    GT.data.resize_(GT_1.size()).copy_(GT_1)



    inputImg_1.data.resize_(images_1.size()).copy_(images_1)
    inputImg_2.data.resize_(by_image.size()).copy_(by_image)

    input_pose_1.data.resize_(po_1.size()).copy_(po_1)
    input_light_1.data.resize_(li_1.size()).copy_(li_1)

    same_pose.data.resize_(same_po.size()).copy_(same_po)
    same_light.data.resize_(same_li.size()).copy_(same_li)
    same_iden.data.resize_(same_id.size()).copy_(same_id)
    netS.zero_grad()

    output_pose_1, output_pose_2, output_light_1, output_light_2, out_f_1, out_f_2, out = netS(inputImg_1, inputImg_2)
    #-----------------mask test area-----------------------------
    # print out.data.type()
    # print GT.data.type()
    # print FinalMask.data.type() same
    # print FinalMask.data.size() 64x3x96x96
    Final_out = FinalMask * out
    Final_GT = FinalMask * GT





    #-----------------mask test area-----------------------------
    # f_1 & f_2 variable
    # same_iden variable
    err_recon = reconstructe_loss(Final_out, Final_GT)
    err_contraI = identity_contrastive_loss(out_f_1, out_f_2, same_iden)
    err_contraP = poss_contrastive_loss(output_pose_1, output_pose_2, same_pose)
    err_contraL = light_contrastive_loss(output_light_1,output_light_2, same_light)
    err_classL = light_class_loss(output_light_1, input_light_1)
    err_classP = pose_class_loss(output_pose_1, input_pose_1)
    # print(err_recon.data.size())
    # print(err_contraL.data.size())
    # print(err_classP.data.size())
    # modify the contrastive loss function to make contrastive loss be 1Lx1L 
    # contrastive loss and Softmax and Loss1 are all requires_grad
    # err_total = 1 * err_recon + 10 * err_contraP + 10 * err_contraI + 10 * err_classP + 20 * err_classL
    # err_total = err_recon + err_contraI + err_contraP + err_contraL + err_classL + err_classP
    err_total = w_r * err_recon
    err_total.backward()
    optimizerS.step()

    #----------------------Visualize-----------
    if(iteration % 200 == 0):

        pose_mtr.add(output_pose_1.data, input_pose_1.data)
        pose_trainacc = pose_mtr.value().diagonal().sum()*1.0/opt.batchSize
        pose_mtr.reset()

        light_mtr.add(output_light_1.data, input_light_1.data)
        light_trainacc = light_mtr.value().diagonal().sum()*1.0/opt.batchSize
        light_mtr.reset()
        #-----------------------------------------
        test(iteration, data_test, loader_test)
    # #pose prediction

    # preds_pose = torch.max(output_pose_1.data, 1)
    # running_corrects += torch.sum(preds == input_pose_1)
    # print('pose_accuracy: %.2f' 
    #         % (running_corrects * 1.0/images.size(0)))
    
    # #light prediction
    # preds_light = torch.max(output_light_1.data, 1)
    # running_corrects_light += torch.sum(preds_light == input_light_1)
    # print('light_accuracy: %.2f' 
    #         % (running_corrects_light * 1.0/images.size(0)))

    print('----------------------------------------')
    print('[%d/%d] Loss_S: %.4f ' %(iteration, opt.niter, err_total.data[0]))
    print('        Reco_S: %.4f ' %(err_recon.data[0]))
    print('        conL_S: %.4f ' %(err_contraL.data[0]))
    print('        conP_S: %.4f ' %(err_contraP.data[0]))
    print('        conI_S: %.4f ' %(err_contraI.data[0]))
    print('        Clas_P: %.4f ' %(err_classP.data[0]))
    print('        Clas_L: %.4f ' %(err_classL.data[0]))


    if(iteration % opt.save_step == 0):
        torch.save(netS.state_dict(), '%s/netS_%d.pth' % (opt.outf,iteration))


