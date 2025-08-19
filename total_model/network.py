import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck
from diffusers.models.attention_processor import Attention
import math

num_classes = 751  # change this depend on your dataset


class MGN(nn.Module):
    def __init__(self):
        super(MGN, self).__init__()

        feats = 256
        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        # self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        # self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        # self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        # self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 8))
        # self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))
        
        self.maxpool_zg_p1 = nn.AdaptiveMaxPool2d((1, 1))
        self.maxpool_zg_p2 = nn.AdaptiveMaxPool2d((1, 1))
        self.maxpool_zg_p3 = nn.AdaptiveMaxPool2d((1, 1))
        self.maxpool_zp2 = nn.AdaptiveMaxPool2d((2, 1))
        self.maxpool_zp3 = nn.AdaptiveMaxPool2d((3, 1))

        self.reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(self.reduction)

        self.fc_id_2048_0 = nn.Linear(feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction(z2_p3).squeeze(dim=3).squeeze(dim=2)

        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3#, p1, p2, p3 #attention 계산시 추가. train 시에는 p1,p2,p3 주석
    
    
    

class Image_adapter(nn.Module):
    """project image embedding to CLIP embedding space"""
    def __init__(self, extractor_dim=2048, hidden_dim=1024, clip_embeddings_dim=2):
        super(Image_adapter, self).__init__()
        self.clip_embeddings_dim = clip_embeddings_dim
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(extractor_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, clip_embeddings_dim * 768),
            torch.nn.LayerNorm(clip_embeddings_dim * 768)
        )
        
    def forward(self, image_embeds):
        projection_embedding = self.proj(image_embeds)
        projection_embedding = projection_embedding.view(-1, self.clip_embeddings_dim, 768)
        return projection_embedding     

class Image_adapter_SDXL(nn.Module):
    """project image embedding to CLIP embedding space"""
    def __init__(self, extractor_dim=2048, hidden_dim=1024, clip_embeddings_dim=2):
        super(Image_adapter_SDXL, self).__init__()
        self.clip_embeddings_dim = clip_embeddings_dim
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(extractor_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, clip_embeddings_dim * 2048),
            torch.nn.LayerNorm(clip_embeddings_dim * 2048)
        )
        
    def forward(self, image_embeds):
        projection_embedding = self.proj(image_embeds)
        projection_embedding = projection_embedding.view(-1, self.clip_embeddings_dim, 2048)
        return projection_embedding     
      
    
class Base_adapter(nn.Module):
    """project image embedding to CLIP embedding space"""
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=768):
        super(Base_adapter, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def blend_embeds(self, image_embeds, text_embeds):
        # image_embeds: (1, 512)
        # text_embeds: (1, 77, 512)
        
        # 이미지 임베딩을 text_embeds와 같은 shape으로 확장
        # (1, 512) -> (1, 1, 512) -> (1, 77, 512)
        #expanded_image_embeds = image_embeds.unsqueeze(1).expand(-1, text_embeds.size(1), -1)
        
        # element-wise 곱셈 수행
        # (1, 77, 512) * (1, 77, 512) -> (1, 77, 512)
        blended = 0.1 * image_embeds + 0.9 * text_embeds
        
        return blended


    def forward(self, image_embeds, text_embeds):
        blended_embedding = self.blend_embeds(image_embeds, text_embeds)
        output_embedding = self.proj(blended_embedding)
        return output_embedding 
    
class Base_adapter_77_768(nn.Module):
    """project image embedding to CLIP embedding space"""
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=768):
        super(Base_adapter_77_768, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def blend_embeds(self, image_embeds, text_embeds):
        # image_embeds: (1, 512)
        # text_embeds: (1, 77, 512)
        
        # 이미지 임베딩을 text_embeds와 같은 shape으로 확장
        # (1, 512) -> (1, 1, 512) -> (1, 77, 512)
        #expanded_image_embeds = image_embeds.unsqueeze(1).expand(-1, text_embeds.size(1), -1)
        
        # element-wise 곱셈 수행
        # (1, 77, 512) * (1, 77, 512) -> (1, 77, 512)
        blended = 0.1 * image_embeds + 0.9 * text_embeds
        
        return blended


    def forward(self, image_embeds, text_embeds):
        blended_embedding = self.blend_embeds(image_embeds, text_embeds)
        #blended_embedding = image_embeds
        output_embedding = self.proj(blended_embedding)
        return output_embedding 


class projector(nn.Module):
    """project image embedding to CLIP embedding space"""
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=768):
        super(projector, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def blend_embeds(self, image_embeds, text_embeds):
        # image_embeds: (1, 512)
        # text_embeds: (1, 77, 512)
        
        # 이미지 임베딩을 text_embeds와 같은 shape으로 확장
        # (1, 512) -> (1, 1, 512) -> (1, 77, 512)
        #expanded_image_embeds = image_embeds.unsqueeze(1).expand(-1, text_embeds.size(1), -1)
        
        # element-wise 곱셈 수행
        # (1, 77, 512) * (1, 77, 512) -> (1, 77, 512)
        blended = 0.1 * image_embeds + 0.9 * text_embeds
        
        return blended


    def forward(self, image_embeds, text_embeds):
        blended_embedding = self.blend_embeds(image_embeds, text_embeds)
        #blended_embedding = image_embeds
        output_embedding = self.proj(blended_embedding)
        return output_embedding 


class aligner(nn.Module):
    """project image embedding to CLIP embedding space"""
    def __init__(self, input_dim=768, output_dim=768):
        super(aligner, self).__init__()
        
        self.aligner = Attention(
            query_dim=768,
            cross_attention_dim=768,
            heads=8,
            dim_head=64,
            dropout=0.,
        )

    def find_character_embeddings(self, captions, text_embeds):
        # 배치 크기만큼의 character 임베딩을 저장할 리스트
        batch_char_embeds = []
        
        # 각 배치에 대해 처리
        for idx, caption in enumerate(captions):
            words = caption.lower().split()
            # "character" 단어의 위치 찾기
            char_positions = [i for i, word in enumerate(words) if "character" in word]
            
            if char_positions:
                # 첫 번째 "character" 단어의 임베딩 사용
                char_pos = char_positions[0]
                char_embed = text_embeds[idx:idx+1, char_pos, :]  # (1, 768) 형태 유지
            else:
                # character 단어가 없는 경우 첫 번째 토큰 사용
                char_embed = text_embeds[idx:idx+1, 0, :]
                
            batch_char_embeds.append(char_embed)
            
        # 배치 차원으로 결합
        return torch.cat(batch_char_embeds, dim=0)

    def forward(self, image_embeds, text_embeds, captions):
        # 배치별 character 임베딩 찾기 (batch_size, 768)
        char_embeddings = self.find_character_embeddings(captions, text_embeds)
        
        # image_embeds의 sequence length에 맞춰서 확장
        seq_len = image_embeds.size(1)
        char_embeddings = char_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        
        # attention 연산 수행
        output = self.aligner(
            hidden_states=char_embeddings,    # query
            encoder_hidden_states=image_embeds # key/value
        )
        
        return output
    
    
class aligner_cls(nn.Module):
    """project image embedding to CLIP embedding space"""
    def __init__(self, input_dim=768, output_dim=768):
        super(aligner, self).__init__()
        
        self.aligner = Attention(
            query_dim=768,
            cross_attention_dim=768,
            heads=8,
            dim_head=64,
            dropout=0.,
        )


    def forward(self, image_embeds, text_embeds, captions):
        # 배치별 character 임베딩 찾기 (batch_size, 768)
        char_embeddings = text_embeds[0]
        
        # image_embeds의 sequence length에 맞춰서 확장
        seq_len = image_embeds.size(1)
        char_embeddings = char_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        
        # attention 연산 수행
        output = self.aligner(
            hidden_states=char_embeddings,    # query
            encoder_hidden_states=image_embeds # key/value
        )
        
        return output
    
    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)
    
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

class Perceiver_Image_adapter_SDXL(nn.Module):
    """Enhanced image adapter using Perceiver architecture with same interface as Image_adapter_SDXL"""
    def __init__(self, extractor_dim=2048, hidden_dim=1024, clip_embeddings_dim=2):
        super(Perceiver_Image_adapter_SDXL, self).__init__()
        self.clip_embeddings_dim = clip_embeddings_dim
        
        # Initial projection of image embeddings
        self.input_proj = nn.Linear(extractor_dim, hidden_dim)
        
        # Latent tokens that will be updated through attention
        self.num_latents = 8
        self.latents = nn.Parameter(torch.randn(1, self.num_latents, hidden_dim))
        
        # Perceiver layers
        self.layers = nn.ModuleList([])
        for _ in range(4):  # 4 layers of perceiver blocks
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=hidden_dim, heads=8, dim_head=64),
                FeedForward(dim=hidden_dim, mult=4)
            ]))
        
        # Final projection to clip embeddings space
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * self.num_latents, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, clip_embeddings_dim * 2048),
            nn.LayerNorm(clip_embeddings_dim * 2048)
        )
        
    def forward(self, image_embeds):
        # Project input embeddings
        x = self.input_proj(image_embeds)
        x = x.unsqueeze(1)  # add sequence dimension
        
        # Initialize latents for this batch
        batch_size = x.shape[0]
        latents = self.latents.expand(batch_size, -1, -1)
        
        # Apply perceiver layers
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        
        # Flatten and project to output space
        latents_flat = latents.reshape(batch_size, -1)
        projection_embedding = self.output_proj(latents_flat)
        
        # Reshape to match expected output format
        projection_embedding = projection_embedding.view(-1, self.clip_embeddings_dim, 2048)
        
        return projection_embedding
    
    
class CLIP_Attention_Image_adapter_SDXL(nn.Module):
    """Image adapter using CLIP image features with MGN as query"""
    def __init__(self, extractor_dim=2048, hidden_dim=1024, clip_embeddings_dim=2):
        super(CLIP_Attention_Image_adapter_SDXL, self).__init__()
        self.clip_embeddings_dim = clip_embeddings_dim
        
        # Initial projection of MGN features
        self.input_proj = nn.Linear(extractor_dim, hidden_dim)
        
        # CLIP 이미지 특성 프로젝션 (CLIP 출력은 512 차원)
        self.clip_image_dim = 512
        self.clip_proj = nn.Linear(self.clip_image_dim, hidden_dim)
        
        # Perceiver layers
        self.layers = nn.ModuleList([])
        for _ in range(4):  # 4 layers of perceiver blocks
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=hidden_dim, heads=8, dim_head=64),
                FeedForward(dim=hidden_dim, mult=4)
            ]))
        
        # Final projection to clip embeddings space
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, clip_embeddings_dim * 2048),
            nn.LayerNorm(clip_embeddings_dim * 2048)
        )
        
    def forward(self, image_embeds, clip_image_features=None):
        # Project MGN embeddings
        mgn_features = self.input_proj(image_embeds)
        mgn_features = mgn_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        batch_size = mgn_features.shape[0]
        
        # 누락된 CLIP 이미지 특성 처리
        if clip_image_features is None:
            clip_image_features = torch.randn(batch_size, self.clip_image_dim).to(mgn_features.device)
        
        # CLIP 이미지 특성 프로젝션 및 시퀀스 차원 추가
        clip_features = self.clip_proj(clip_image_features)
        clip_features = clip_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # MGN 특성을 attention의 latent(query)로 사용 - 핵심 변경점!
        attention_features = mgn_features
        
        # Apply perceiver layers (이제 MGN 특성이 query, CLIP 특성이 key/value 소스)
        for attn, ff in self.layers:
            # 순서 변경: clip_features가 첫 번째 인자(key/value 소스), attention_features가 두 번째 인자(query 소스)
            attention_features = attn(clip_features, attention_features) + attention_features
            attention_features = ff(attention_features) + attention_features
        
        # 최종 출력 처리
        pooled_features = attention_features.squeeze(1)  # [batch_size, hidden_dim]
        projection_embedding = self.output_proj(pooled_features)
        
        # 원하는 출력 형태로 재구성
        projection_embedding = projection_embedding.view(-1, self.clip_embeddings_dim, 2048)
        
        return projection_embedding
    
    
    