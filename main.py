import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib
from PIL import Image
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torch.optim import lr_scheduler

import wandb
from transformers import CLIPProcessor, CLIPModel

from opt import opt
from data import Data
from network import MGN, Image_adapter
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking

import pdb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Main():
    def __init__(self, extractor, image_adapter):
        data = Data()
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.extractor = extractor
        self.loss = Loss()
        self.optimizer = get_optimizer(self.extractor)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)
        self.image_adapter = image_adapter
        self.text_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # load pretrained weight
        if hasattr(opt, 'weight') and opt.weight:
            print(f"Loading pretrained weights from {opt.weight}")
            self.extractor.load_state_dict(torch.load(opt.weight))
            
        # wandb 초기화
        wandb.init(
            project="Controlnet",
            config=vars(opt),  # opt의 모든 속성을 config로 추가
            name=f"experiment_{opt.mode}_lr_{opt.lr}_epoch_{opt.epoch}_{opt.batchid} * {opt.batchimage}"
        )

    def train(self, epoch):
        self.extractor.eval()  # freeze ID extractor
        self.image_adapter.train()  # train image adapter
        self.text_encoder.eval()

        total_loss = 0

        for batch, (inputs, labels, captions) in enumerate(self.train_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            self.optimizer.zero_grad()

            # Image embedding
            outputs = self.extractor(inputs)
            embedding_feature = outputs[0]  # (batch, 2048) embedding 추출

            projected_image_embedding = self.image_adapter(embedding_feature)  # (batch, 1024) embedding

            # Text embedding
            with torch.no_grad():
                # 캡션 텍스트를 전처리
                text_inputs = self.processor(text=captions, return_tensors="pt", padding=True).to('cuda')
                text_embedding = self.text_encoder.get_text_features(**text_inputs)

            # 손실 계산 및 역전파
            loss, Triplet_Loss, CrossEntropy_Loss, Contrastive_Loss = self.loss(outputs, labels, projected_image_embedding, text_embedding)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # wandb에 배치별 손실 기록
            wandb.log({"batch_loss": loss.item(),
                       "Triplet_Loss": Triplet_Loss.item(),
                       "CrossEntropy_Loss": CrossEntropy_Loss.item(),
                       "Contrastive_Loss": Contrastive_Loss.item()})

        # 에포크별 평균 손실을 기록
        avg_loss = total_loss / len(self.train_loader)
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch})

    def evaluate(self):
        self.extractor.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.extractor, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.extractor, tqdm(self.test_loader)).numpy()


if __name__ == '__main__':
    extractor = MGN().to('cuda')
    image_adapter = Image_adapter().to('cuda')
    main = Main(extractor, image_adapter)

    if opt.mode == 'train':
        for epoch in range(1, opt.epoch + 1):
            print('\nepoch', epoch)
            
            main.train(epoch)
            if epoch % 500 == 0:
                print('\nstart evaluate')
                os.makedirs('./weights/256_limit_resume_adapter', exist_ok=True)
                torch.save(extractor.state_dict(), ('./weights/256_limit_resume_adapter/extractor_{}.pt'.format(epoch)))
                torch.save(image_adapter.state_dict(), ('./weights/256_limit_resume_adapter/image_adapter_{}.pt'.format(epoch)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        extractor.load_state_dict(torch.load(opt.weight))
        main.evaluate()
