

from __future__ import print_function

import matplotlib; matplotlib.use('Agg')
import os
import os.path as osp
import argparse

from train import train 
from test import test
from test_beam import test_beam

# argument 설정
parser = argparse.ArgumentParser(description='PyTorch Convolutional Image Captioning Model')

parser.add_argument('model_dir', help='output directory to save models & results')

parser.add_argument('-g', '--gpu', type=int, default=0,                    help='gpu device id')

parser.add_argument('--coco_root', type=str, default= './data/coco/',                    help='directory containing coco dataset train2014, val2014, & annotations')

parser.add_argument('-t', '--is_train', type=int, default=1,                    help='use 1 to train model')

parser.add_argument('-e', '--epochs', type=int, default=15,                    help='number of training epochs')

parser.add_argument('-b', '--batchsize', type=int, default=20,                    help='number of images per training batch')

parser.add_argument('-c', '--ncap_per_img', type=int, default=5,                    help='ground-truth captions per image in training batch')

parser.add_argument('-n', '--num_layers', type=int, default=3,                    help='depth of convcap network')

parser.add_argument('-m', '--nthreads', type=int, default=4,                    help='pytorch data loader threads')

# parser.add_argument('-ft', '--finetune_after', type=int, default=8,\
#                     help='epochs after which vgg16 is fine-tuned')

parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5,                    help='learning rate for convcap')

parser.add_argument('-st', '--lr_step_size', type=int, default=15,                    help='epochs to decay learning rate after')

parser.add_argument('-sc', '--score_select', type=str, default='CIDEr',                    help='metric to pick best model')

parser.add_argument('--beam_size', type=int, default=1,                     help='beam size to use for test') 

parser.add_argument('--attention', dest='attention', action='store_true',                     help='Use this for convcap with attention (by default set)')

parser.add_argument('--no-attention', dest='attention', action='store_false',                     help='Use this for convcap without attention')


parser.set_defaults(attention=True)
args, _ = parser.parse_known_args()
args.finetune_after = 8
args.model_dir = 'output'


import os
import os.path as osp
import argparse
import numpy as np 
import json
import time
 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models                                                                     

from coco_loader import coco_loader
from convcap import convcap
from vggfeats import Vgg16Feats
from tqdm import tqdm 
from test import test 


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if (args.is_train == 1):
    print('train')

t_start = time.time()
#coco_loader를 사용하여 훈련데이터 로드
train_data = coco_loader(args.coco_root, split='train', ncap_per_img=args.ncap_per_img)
print('[DEBUG] Loading train data ... %f secs' % (time.time() - t_start))

#데이터로더 생성
train_data_loader = DataLoader(dataset=train_data, num_workers=0, batch_size=args.batchsize,
                               shuffle=True, drop_last=True)

#VGG16 모델 로드
model_imgcnn = Vgg16Feats()
model_imgcnn.cuda()
model_imgcnn.train(True)


#Convcap model
#캡션을 생성해주는 Convcap model 불러옴
model_convcap = convcap(train_data.numwords, args.num_layers, is_attention=args.attention)
model_convcap.cuda()
model_convcap.train(True)

#optimizer 와 scheduler 설정
optimizer = optim.RMSprop(model_convcap.parameters(), lr=args.learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=.1)
img_optimizer = None

#하이퍼 파라미터 설정
batchsize = args.batchsize
ncap_per_img = args.ncap_per_img
batchsize_cap = batchsize*ncap_per_img
max_tokens = train_data.max_tokens
nbatches = np.int_(np.floor((len(train_data.ids)*1.)/batchsize)) 
bestscore = .0


def repeat_img_per_cap(imgsfeats, imgsfc7, ncap_per_img):
    batchsize, featdim, feat_h, feat_w = imgsfeats.size()
    batchsize_cap = batchsize*ncap_per_img
    imgsfeats = imgsfeats.unsqueeze(1).expand(batchsize, ncap_per_img, featdim, feat_h, feat_w)
    imgsfeats = imgsfeats.contiguous().view(batchsize_cap, featdim, feat_h, feat_w)
    
    batchsize, featdim = imgsfc7.size()
    batchsize_cap = batchsize*ncap_per_img
    imgsfc7 = imgsfc7.unsqueeze(1).expand(batchsize, ncap_per_img, featdim)
    imgsfc7 = imgsfc7.contiguous().view(batchsize_cap, featdim)
    
    return imgsfeats, imgsfc7


# 시간상 15번 실행했습니다.
for epoch in range(15): 
    loss_train = 0.

    if(epoch == args.finetune_after):
        img_optimizer = optim.RMSprop(model_imgcnn.parameters(), lr=1e-5)
        img_scheduler = lr_scheduler.StepLR(img_optimizer, step_size=args.lr_step_size, gamma=.1)

    scheduler.step()    
    if(img_optimizer):
        img_scheduler.step()

    #Train data loader 로드
    #이미지, 캡션, index로 인코딩된 캡션, 마스크
    for batch_idx, (imgs, captions, wordclass, mask, _) in         tqdm(enumerate(train_data_loader), total=nbatches):
        
        
        imgs = imgs.view(batchsize, 3, 224, 224)
        wordclass = wordclass.view(batchsize_cap, max_tokens)
        mask = mask.view(batchsize_cap, max_tokens)
        
        #Training data 
        imgs_v = Variable(imgs).cuda()
        wordclass_v = Variable(wordclass).cuda()

        optimizer.zero_grad()
        if(img_optimizer):
            img_optimizer.zero_grad() 
        
        #VGG16에 INPUT하여 image feature 추출
        imgsfeats, imgsfc7 = model_imgcnn(imgs_v)
        
        #image 와 caption이 1:1이 되도록 함
        imgsfeats, imgsfc7 = repeat_img_per_cap(imgsfeats, imgsfc7, ncap_per_img)
        _, _, feat_h, feat_w = imgsfeats.size()

        
        if(args.attention == True):
            wordact, attn = model_convcap(imgsfeats, imgsfc7, wordclass_v)
            attn = attn.view(batchsize_cap, max_tokens, feat_h, feat_w)
        else:
            wordact, _ = model_convcap(imgsfeats, imgsfc7, wordclass_v)

        wordact = wordact[:,:,:-1]
        wordclass_v = wordclass_v[:,1:]
        mask = mask[:,1:].contiguous()

        wordact_t = wordact.permute(0, 2, 1).contiguous().view(        batchsize_cap*(max_tokens-1), -1)
        wordclass_t = wordclass_v.contiguous().view(        batchsize_cap*(max_tokens-1), 1)

        maskids = torch.nonzero(mask.view(-1)).numpy().reshape(-1)

        if(args.attention == True):
        #Cross-entropy loss 계산
            loss = F.cross_entropy(wordact_t[maskids, ...],               wordclass_t[maskids, ...].contiguous().view(maskids.shape[0]))               + (torch.sum(torch.pow(1. - torch.sum(attn, 1), 2)))              /(batchsize_cap*feat_h*feat_w)
        else:
            loss = F.cross_entropy(wordact_t[maskids, ...],               wordclass_t[maskids, ...].contiguous().view(maskids.shape[0]))

        loss_train = loss_train + loss.data

        loss.backward()

        optimizer.step()
        if(img_optimizer):
            img_optimizer.step()

    loss_train = (loss_train*1.)/(batch_idx)
    print('[DEBUG] Training epoch %d has loss %f' % (epoch, loss_train))

    modelfn = osp.join(args.model_dir, 'model.pth')

    if(img_optimizer):
        img_optimizer_dict = img_optimizer.state_dict()
    else:
        img_optimizer_dict = None

    torch.save({
        'epoch': epoch,
        'state_dict': model_convcap.state_dict(),
        'img_state_dict': model_imgcnn.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'img_optimizer' : img_optimizer_dict,
      }, modelfn)

    #Run on validation and obtain score
    scores = test(args, 'val', model_convcap=model_convcap, model_imgcnn=model_imgcnn)
    score = scores[0][args.score_select]

    if(score > bestscore):
        bestscore = score
        print('[DEBUG] Saving model at epoch %d with %s score of %f'        % (epoch, args.score_select, score))
        bestmodelfn = osp.join(args.model_dir, 'bestmodel.pth')
        os.system('cp %s %s' % (modelfn, bestmodelfn))

#훈련 종료
print("Training cycle end")
bestmodelfn = osp.join(args.model_dir, 'bestmodel.pth')


# 스코어 산출 TEST
if (osp.exists(bestmodelfn)):
    print('if (osp.exists(bestmodelfn)):')
    
    if (args.beam_size == 1):
        print('if (args.beam_size == 1):')
        scores = test(args, 'test', modelfn=bestmodelfn)
    else:
        print('else:')
        scores = test_beam(args, 'test', modelfn=bestmodelfn)
        
    print('TEST set scores')
    for k, v in scores[0].items():
        print('%s: %f' % (k, v))
else:
    print('2 else')
    raise Exception('No checkpoint found %s' % bestmodelfn)


scores[0].items()
