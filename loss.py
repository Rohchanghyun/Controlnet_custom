from torch.nn import CrossEntropyLoss
from torch.nn.modules import loss
from utils.TripletLoss import TripletLoss
from torch import nn
import torch


class Loss(loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()
        # 추가된 contrastive loss 계산을 위한 온도 파라미터
        self.temperature = 0.07
        self.cross_entropy_loss = CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=1.2)
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def contrastive_loss(self, image_embeds, text_embeds):
        # 이미지와 텍스트 임베딩 간의 코사인 유사도 계산
        logits = self.cosine_similarity(image_embeds.unsqueeze(1), text_embeds.unsqueeze(0)) / self.temperature

        # 대각선 요소가 양의 쌍 (positive pair)
        labels = torch.arange(logits.size(0)).to(image_embeds.device)

        # CrossEntropyLoss로 contrastive loss 계산
        contrastive_loss = self.cross_entropy_loss(logits, labels)
        return contrastive_loss

    def forward(self, outputs, labels, image_embeds, text_embeds):
        # Triplet Loss 계산
        Triplet_Loss = [self.triplet_loss(output, labels) for output in outputs[1:4]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        # CrossEntropy Loss 계산
        CrossEntropy_Loss = [self.cross_entropy_loss(output, labels) for output in outputs[4:]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        # Contrastive Loss 계산
        Contrastive_Loss = self.contrastive_loss(image_embeds, text_embeds)

        # 최종 손실 함수 계산 (Triplet Loss + CrossEntropy Loss + Contrastive Loss)
        loss_sum = 0.1 * Triplet_Loss +  0.1 * CrossEntropy_Loss + Contrastive_Loss

        print('total loss: {:.2f}, Triplet_Loss: {:.2f}, CrossEntropy_Loss: {:.2f}, Contrastive_Loss: {:.2f}'.format(
            loss_sum.item(),
            Triplet_Loss.item(),
            CrossEntropy_Loss.item(),
            Contrastive_Loss.item()))
        return loss_sum, Triplet_Loss, CrossEntropy_Loss, Contrastive_Loss
