from torch.nn import CrossEntropyLoss
from torch.nn.modules import loss
from utils.TripletLoss import TripletLoss
from torch import nn
import torch
import pdb

class Loss(loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()
        # 추가된 contrastive loss 계산을 위한 온도 파라미터
        self.temperature = 0.20
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
        self.triplet_loss = TripletLoss(margin=2.0)
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def contrastive_loss(self, image_embeds, text_embeds):
        # 이미지와 텍스트 임베딩 간의 코사인 유사도 계산
        logits = self.cosine_similarity(image_embeds.unsqueeze(1), text_embeds.unsqueeze(0)) / self.temperature
        #logits = self.cosine_similarity(image_embeds, text_embeds) / self.temperature
        # 대각선 요소가 양의 쌍 (positive pair)
        labels = torch.arange(logits.size(0)).to(image_embeds.device)

        # CrossEntropyLoss로 contrastive loss 계산
        contrastive_loss = self.cross_entropy_loss(logits, labels)
        return contrastive_loss
    
    def contrastive_loss_77(self, image_embeds, text_embeds):
        # image_embeds shape: (B, 77, 512)
        # text_embeds shape: (B, 77, 512)
        
        batch_size = image_embeds.size(0)
        total_loss = 0
        
        # 배치의 각 항목별로 독립적으로 처리
        for i in range(batch_size):
            # 현재 배치의 이미지와 텍스트 임베딩 추출
            curr_image_embeds = image_embeds[i]  # shape: (77, 512)
            curr_text_embeds = text_embeds[i]    # shape: (77, 512)
            
            # 현재 이미지-텍스트 쌍의 코사인 유사도 계산
            logits = self.cosine_similarity(curr_image_embeds.unsqueeze(1), 
                                        curr_text_embeds.unsqueeze(0)) / self.temperature
            # logits shape: (77, 77)
            
            # 현재 쌍의 레이블 생성
            labels = torch.arange(logits.size(0)).to(image_embeds.device)
            
            # 현재 쌍의 loss 계산
            curr_loss = self.cross_entropy_loss(logits, labels)
            total_loss += curr_loss
        
        # 평균 loss 반환
        return total_loss / batch_size

    def forward(self, outputs, labels, image_embeds, text_embeds):
        # Triplet Loss 계산
        Triplet_Loss = [self.triplet_loss(outputs, labels) for outputs in outputs[1:4]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)
        

        # CrossEntropy Loss 계산
        CrossEntropy_Loss = [self.cross_entropy_loss(outputs, labels) for outputs in outputs[4:]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        # Contrastive Loss 계산
        Contrastive_Loss = self.contrastive_loss(image_embeds, text_embeds)
        #Contrastive_Loss_77 = self.contrastive_loss_77(image_embeds, text_embeds)

        # 최종 손실 함수 계산 (Triplet Loss + CrossEntropy Loss + Contrastive Loss)
        #loss_sum = Triplet_Loss + CrossEntropy_Loss + Contrastive_Loss
        loss_sum = 0.5 * Triplet_Loss + 0.5 * CrossEntropy_Loss + 2 * Contrastive_Loss

        print('total loss: {:.2f}, Triplet_Loss: {:.2f}, CrossEntropy_Loss: {:.2f}, Contrastive_Loss: {:.2f}'.format(
            loss_sum.item(),
            Triplet_Loss.item(),
            CrossEntropy_Loss.item(),
            Contrastive_Loss.item()))
        return loss_sum, Triplet_Loss, CrossEntropy_Loss, Contrastive_Loss
