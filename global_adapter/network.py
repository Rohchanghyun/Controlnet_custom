import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck
from diffusers.models.attention_processor import Attention
import math
import xformers
import torch.nn.functional as F

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
            torch.nn.Linear(hidden_dim, clip_embeddings_dim * 1024),
            torch.nn.LayerNorm(clip_embeddings_dim * 1024)
        )
        
    def forward(self, image_embeds):
        projection_embedding = self.proj(image_embeds)
        projection_embedding = projection_embedding.view(-1, self.clip_embeddings_dim, 1024)
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

    def forward(self, x, latents, return_attention=False):
        x = self.norm1(x) # (2,1,1024)
        latents = self.norm2(latents) # (2,1,1024)

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

        if return_attention:
            return self.to_out(out), weight
        else:
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
        
        # Perceiver cross attention (1회)
        self.cross_attn = PerceiverAttention(dim=hidden_dim, heads=8, dim_head=64)
        self.cross_ff = FeedForward(dim=hidden_dim, mult=4)
        
        # Perceiver self attention (3회)
        self.self_layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=hidden_dim, heads=8, dim_head=64),
                FeedForward(dim=hidden_dim, mult=4)
            ]) for _ in range(3)
        ])
        
        # Final projection to clip embeddings space
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, clip_embeddings_dim * 2048),
            nn.LayerNorm(clip_embeddings_dim * 2048)
        )
        
    def forward(self, clip_feature, embedding_feature, return_attention=False, return_debug_shapes=False):
        debug_shapes = []
        
        # Project MGN embeddings
        mgn_features = self.input_proj(embedding_feature)
        mgn_features = mgn_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        batch_size = mgn_features.shape[0]
        
        # CLIP 이미지 특성 시퀀스 차원 추가
        clip_features = clip_feature.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        attention_maps = []
        
        # 1. Cross attention (MGN이 query, CLIP이 key/value)
        attn_out, attn_map = self.cross_attn(clip_features, mgn_features, return_attention=True)
        attention_maps.append(attn_map)
        attention_features = attn_out + mgn_features
        attention_features = self.cross_ff(attention_features) + attention_features
        
        # 2. Self attention (3회, latent끼리)
        for attn, ff in self.self_layers:
            latents = attn(attention_features, attention_features) + attention_features
            latents = ff(latents) + latents
        
        # 최종 출력 처리
        pooled_features = latents.squeeze(1)  # [batch_size, hidden_dim]
        projection_embedding = self.output_proj(pooled_features)
        
        # 원하는 출력 형태로 재구성
        projection_embedding = projection_embedding.view(-1, self.clip_embeddings_dim, 2048)
        
        if return_attention and return_debug_shapes:
            return projection_embedding, attention_maps, debug_shapes
        elif return_attention:
            return projection_embedding, attention_maps
        else:
            return projection_embedding


class Transformer_CLIP_Attention_Image_adapter_SDXL(nn.Module):
    """공식 Transformer 블록을 활용한 CLIP+MGN 어댑터"""
    def __init__(self, extractor_dim=2048, hidden_dim=1024, clip_embeddings_dim=2, nhead=8):
        super().__init__()
        self.clip_embeddings_dim = clip_embeddings_dim

        # MGN 임베딩 프로젝션
        self.input_proj = nn.Linear(extractor_dim, hidden_dim)
        # CLIP 임베딩 프로젝션 (차원이 다를 경우)
        self.clip_proj = nn.Linear(extractor_dim, hidden_dim)

        # Cross attention: 쿼리=MGN, 키/밸류=CLIP
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        self.cross_ff = FeedForward(hidden_dim, mult=4)

        # Self attention 3회
        self.self_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, batch_first=True, activation='gelu')
            for _ in range(3)
        ])

        # 최종 프로젝션
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, clip_embeddings_dim * 2048),
            nn.LayerNorm(clip_embeddings_dim * 2048)
        )

    def forward(self, image_embeds, clip_image_features, return_attention=False):
        # 1. 프로젝션
        mgn_features = self.input_proj(image_embeds).unsqueeze(1)  # [B, 1, H]
        clip_features = self.clip_proj(clip_image_features).unsqueeze(1)  # [B, 1, H]

        # 2. Cross attention (쿼리: MGN, 키/밸류: CLIP)
        # nn.MultiheadAttention은 (N, S, E) 입력, 쿼리/키/밸류 순서
        attn_out, attn_weights = self.cross_attn(mgn_features, clip_features, clip_features, need_weights=return_attention)
        attention_features = attn_out + mgn_features
        attention_features = self.cross_ff(attention_features) + attention_features

        # 3. Self attention 3회
        for layer in self.self_layers:
            attention_features = layer(attention_features)

        # 4. 출력 변환
        pooled_features = attention_features.squeeze(1)
        projection_embedding = self.output_proj(pooled_features)
        projection_embedding = projection_embedding.view(-1, self.clip_embeddings_dim, 2048)

        if return_attention:
            # cross attention만 반환 (self attention은 공식 블록에서 반환 불가)
            return projection_embedding, attn_weights
        else:
            return projection_embedding


class SimpleCrossAttention1D(nn.Module):
    def __init__(self, query_dim = 1024, kv_dim = 1024, inner_dim = 2048, dropout=0):
        super().__init__()
        self.scale = 1 / math.sqrt(1024)  # 1024: dim of key/value
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential( nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def forward(self, query, key, value, return_attention=False):
        # query: [B, 1, 2048], key/value: [B, 77, 1024]

        q = self.to_q(query)     # B*1*(inner_dim)
        k = self.to_k(key)   # B*M*(inner_dim)
        v = self.to_v(value) # B*M*(inner_dim)

        B, N, _ = q.shape
        _, M, _ = k.shape

        # 3. Attention map: [B, N, M]
        attn = torch.matmul(q, k.transpose(1, 2)) * self.scale  # [B, 1, 77]
        attn = torch.softmax(attn, dim=-1)

        # 4. Weighted sum: [B, 1024, 1]
        out = torch.matmul(attn, v)  # [B, 1024, 1]

        # 5. 다시 [B, 1, 1024]로 transpose
        out = out.transpose(1, 2)  # [B, 1, 1024]
        
        

        if return_attention:
            return out, attn
        else:
            return out
        

class SimpleSelfAttention1D(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x, return_attention=False):
        # x: [B, 1, 1024]
        # nn.MultiheadAttention expects [B, seq_len, embed_dim], so transpose to [B, 1024, 1] then back
        x_in = x  # [B, 1, 1024]
        x = x.transpose(1, 2)  # [B, 1024, 1]
        # MultiheadAttention expects [B, seq_len, embed_dim] where embed_dim=1, but we want embed_dim=1024
        # So we treat the 1024 as embed_dim, seq_len=1
        # Therefore, input shape should be [B, 1, 1024]
        # Actually, no need to transpose, just use as is:
        attn_out, attn_weights = self.self_attn(x_in, x_in, x_in, need_weights=return_attention)
        # attn_out: [B, 1, 1024], attn_weights: [B, 1, 1]
        if return_attention:
            return attn_out, attn_weights
        else:
            return attn_out


class SimpleSelfAttention1D_Transpose(nn.Module):
    """
    입력: [B, 1024, 1]
    출력: [B, 1, 1024]
    attention map: [B, 1024, 1024]
    """
    def __init__(self):
        super().__init__()
        self.scale = 1 / math.sqrt(1)  # 마지막 차원이 1이므로

    def forward(self, x, return_attention=False):
        # x: [B, 1024, 1]
        q = x  # [B, 1024, 1]
        k = x  # [B, 1024, 1]
        v = x  # [B, 1024, 1]

        # Attention map: [B, 1024, 1024]
        attn = torch.matmul(q, k.transpose(1, 2)) * self.scale  # [B, 1024, 1024]
        attn = torch.softmax(attn, dim=-1)

        # Weighted sum: [B, 1024, 1]
        out = torch.matmul(attn, v)  # [B, 1024, 1]

        # 원하는 출력: [B, 1, 1024]
        out = out.transpose(1, 2)  # [B, 1, 1024]

        if return_attention:
            return out, attn
        else:
            return out

class SimpleCrossSelfAttentionStack1D(nn.Module):
    def __init__(self, extractor_dim=2048, hidden_dim=1024, clip_embeddings_dim=2):
        super().__init__()
        self.clip_embeddings_dim = clip_embeddings_dim
        self.input_proj = nn.Linear(extractor_dim, hidden_dim)
        self.cross_attn = SimpleCrossAttention1D()
        self.cross_ff = FeedForward(hidden_dim, mult=4)
        self.self_attn_layers = nn.ModuleList([SimpleSelfAttention1D(embed_dim=hidden_dim, num_heads=1) for _ in range(3)])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, clip_embeddings_dim * 2048),
            nn.LayerNorm(clip_embeddings_dim * 2048)
        )

    def forward(self, embedding_feature, clip_feature, return_attention=False, return_debug_shapes=False):
        debug_shapes = []

        # 1. Project MGN embeddings
        mgn_features = self.input_proj(embedding_feature)  # [B, hidden_dim]
        mgn_features = mgn_features.unsqueeze(1)           # [B, 1, hidden_dim]
        clip_features = clip_feature.unsqueeze(1)           # [B, 1, hidden_dim]

        attention_maps = []

        # 2. Cross attention (MGN이 query, CLIP이 key/value)
        attn_out, attn_map = self.cross_attn(mgn_features, clip_features, clip_features, return_attention=True)
        attention_maps.append(attn_map)
        attention_features = attn_out + mgn_features
        attention_features = self.cross_ff(attention_features) + attention_features

        # 3. Self attention (3회)
        for self_attn in self.self_attn_layers:
            attn_out, attn_map = self_attn(attention_features, return_attention=True)
            attention_maps.append(attn_map)
            attention_features = attn_out + attention_features

        # 4. 최종 출력 처리
        pooled_features = attention_features.squeeze(1)  # [B, hidden_dim]
        projection_embedding = self.output_proj(pooled_features)
        projection_embedding = projection_embedding.view(-1, self.clip_embeddings_dim, 2048)

        if return_attention and return_debug_shapes:
            return projection_embedding, attention_maps, debug_shapes
        elif return_attention:
            return projection_embedding, attention_maps
        else:
            return projection_embedding

class SimpleSelfAttention1D_Transpose(nn.Module):
    """
    입력: [B, 1024, 1]
    출력: [B, 1024, 1] (혹은 [B, 1, 1024]로 transpose)
    attention map: [B, 1024, 1024]
    """
    def __init__(self, num_heads=1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=1, num_heads=num_heads, batch_first=True)

    def forward(self, x, return_attention=False):
        
        # x: [B, 1024, 1]
        out, attn_weights = self.self_attn(x, x, x, need_weights=return_attention)
        # attn_out: [B, 1024, 1], attn_weights: [B, 1024, 1024]
        # 필요하다면 transpose로 [B, 1, 1024]로 변환
        if return_attention:
            return out, attn_weights
        else:
            return out
        
        
class SimpleCrossSelfAttentionStack1D_transpose(nn.Module):
    def __init__(self, extractor_dim=2048, hidden_dim=1024, clip_embeddings_dim=2):
        super().__init__()
        self.clip_embeddings_dim = clip_embeddings_dim
        self.input_proj = nn.Linear(extractor_dim, hidden_dim)
        self.self_attn_layers = nn.ModuleList([SelfAttention() for _ in range(3)])
        self.cross_attn = CrossAttention()
        self.cross_ff = FeedForward(hidden_dim, mult=4)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, clip_embeddings_dim * 2048),
            nn.LayerNorm(clip_embeddings_dim * 2048)
        )

    def forward(self, embedding_feature, clip_feature, return_attention=False, return_debug_shapes=False):
        debug_shapes = []

        # 1. Project MGN embeddings
        mgn_features = self.input_proj(embedding_feature)  # [B, hidden_dim]
        mgn_features = mgn_features.unsqueeze(1)           # [B, 1, hidden_dim]

        attention_maps = []

        # 2. Cross attention (MGN이 query, CLIP이 key/value)
        attn_out, attn_map = self.cross_attn(mgn_features, clip_feature, clip_feature, return_attention=True)
        print(f'attn_out: {attn_out.shape}, attn_map: {attn_map.shape}')
        print(f'attn_map.shape: {attn_map.shape}')
        attention_maps.append(attn_map)
        attention_features = attn_out + mgn_features
        attention_features = self.cross_ff(attention_features) + attention_features

        # 3. Self attention (3회)
        # self_attn = SimpleSelfAttention1D_Transpose(num_heads=1)
        # attn_out, attn_map = self_attn(attention_features, return_attention=True)
        for self_attn in self.self_attn_layers:
            attn_out, attn_map = self_attn(attention_features, return_attention=True)
            attention_maps.append(attn_map)
            attention_features = attn_out + attention_features
        # 4. 최종 출력 처리
        pooled_features = attention_features.squeeze(1)  # [B, hidden_dim]
        projection_embedding = self.output_proj(pooled_features)
        projection_embedding = projection_embedding.view(-1, self.clip_embeddings_dim, 2048)

        if return_attention and return_debug_shapes:
            return projection_embedding, attention_maps, debug_shapes
        elif return_attention:
            return projection_embedding, attention_maps
        else:
            return projection_embedding
        
class SingleCondition_ImageAdapter(nn.Module):
    """
    condition embedding이 하나만 들어올 때 사용하는 FeedForward 기반 어댑터.
    GlobalFuser의 foreground branch와 동일한 구조.
    """
    def __init__(self, in_dim=2048, channel_mult=[2, 4]):
        super().__init__()
        dim_out1 = in_dim * channel_mult[0]
        dim_out2 = in_dim * channel_mult[1]
        self.in_dim = in_dim
        self.channel_mult = channel_mult

        self.ff1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim_out1, bias=False),
            nn.GELU(),
            nn.Linear(dim_out1, dim_out1, bias=False),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim_out1),
            nn.Linear(dim_out1, dim_out2, bias=False),
            nn.GELU(),
            nn.Linear(dim_out2, dim_out2, bias=False),
        )

    def forward(self, cond_emb):
        # cond_emb: (batch, in_dim)
        x = self.ff1(cond_emb)
        x = self.ff2(x)
        # (batch, n*d) -> (batch, n, d)
        x = x.view(x.shape[0], self.channel_mult[-1], self.in_dim)
        return x  # (batch, n, d)
    
class Residual_SingleCondition_ImageAdapter(nn.Module):
    """
    FeedForward 기반 어댑터 + 입력 residual connection 추가 버전.
    입력 차원과 출력 차원이 동일할 때만 residual이 정상적으로 동작합니다.
    """
    def __init__(self, in_dim=2048, channel_mult=[2, 4]):
        super().__init__()
        dim_out1 = in_dim * channel_mult[0]
        dim_out2 = in_dim * channel_mult[1]
        self.in_dim = in_dim
        self.channel_mult = channel_mult

        self.ff1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim_out1, bias=False),
            nn.GELU(),
            nn.Linear(dim_out1, dim_out1, bias=False),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim_out1),
            nn.Linear(dim_out1, dim_out2, bias=False),
            nn.GELU(),
            nn.Linear(dim_out2, dim_out2, bias=False),
        )

        # residual projection (입력과 출력 차원이 다르면 맞춰줌)
        self.residual_proj = nn.Linear(in_dim, in_dim * channel_mult[-1], bias=False) if channel_mult[-1] != 1 else nn.Identity()

    def forward(self, cond_emb):
        # cond_emb: (batch, in_dim)
        x = self.ff1(cond_emb)
        x = self.ff2(x)
        # (batch, n*d) -> (batch, n, d)
        x = x.view(x.shape[0], self.channel_mult[-1], self.in_dim)
        # residual connection
        residual = self.residual_proj(cond_emb).view(x.shape[0], self.channel_mult[-1], self.in_dim)
        x = x + residual
        return x  # (batch, n, d)
    
class ResidualEachBlock_SingleCondition_ImageAdapter(nn.Module):
    """
    FeedForward 각 블록(ff1, ff2)마다 residual connection을 적용한 어댑터.
    """
    def __init__(self, in_dim=2048, channel_mult=[2, 4]):
        super().__init__()
        dim_out1 = in_dim * channel_mult[0]
        dim_out2 = in_dim * channel_mult[1]
        self.in_dim = in_dim
        self.channel_mult = channel_mult

        self.ff1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim_out1, bias=False),
            nn.GELU(),
            nn.Linear(dim_out1, dim_out1, bias=False),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim_out1),
            nn.Linear(dim_out1, dim_out2, bias=False),
            nn.GELU(),
            nn.Linear(dim_out2, dim_out2, bias=False),
        )

        # ff1, ff2 residual projection
        self.residual_proj1 = nn.Linear(in_dim, dim_out1, bias=False) if dim_out1 != in_dim else nn.Identity()
        self.residual_proj2 = nn.Linear(dim_out1, dim_out2, bias=False) if dim_out2 != dim_out1 else nn.Identity()

    def forward(self, cond_emb):
        # ff1 + residual
        x1 = self.ff1(cond_emb)
        res1 = self.residual_proj1(cond_emb)
        x1 = x1 + res1

        # ff2 + residual
        x2 = self.ff2(x1)
        res2 = self.residual_proj2(x1)
        x2 = x2 + res2

        # (batch, n*d) -> (batch, n, d)
        x2 = x2.view(x2.shape[0], self.channel_mult[-1], self.in_dim)
        return x2  # (batch, n, d)
    
class ResidualEachBlock_SingleCondition_ImageAdapter_4Layers(nn.Module):
    """
    FeedForward 각 블록(ff1, ff2, ff3, ff4)마다 residual connection을 적용한 어댑터.
    channel_mult를 4개로 받아서 차원을 점진적으로 확장합니다.
    """
    def __init__(self, in_dim=2048, channel_mult=[2, 4, 8]):
        super().__init__()
        dim_out1 = in_dim * channel_mult[0]
        dim_out2 = in_dim * channel_mult[1]
        dim_out3 = in_dim * channel_mult[2]
        self.in_dim = in_dim
        self.channel_mult = channel_mult

        self.ff1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim_out1, bias=False),
            nn.GELU(),
            nn.Linear(dim_out1, dim_out1, bias=False),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim_out1),
            nn.Linear(dim_out1, dim_out2, bias=False),
            nn.GELU(),
            nn.Linear(dim_out2, dim_out2, bias=False),
        )
        self.ff3 = nn.Sequential(
            nn.LayerNorm(dim_out2),
            nn.Linear(dim_out2, dim_out3, bias=False),
            nn.GELU(),
            nn.Linear(dim_out3, dim_out3, bias=False),
        )

        # 각 블록별 residual projection
        self.residual_proj1 = nn.Linear(in_dim, dim_out1, bias=False) if dim_out1 != in_dim else nn.Identity()
        self.residual_proj2 = nn.Linear(dim_out1, dim_out2, bias=False) if dim_out2 != dim_out1 else nn.Identity()
        self.residual_proj3 = nn.Linear(dim_out2, dim_out3, bias=False) if dim_out3 != dim_out2 else nn.Identity()

    def forward(self, cond_emb):
        # ff1 + residual
        x1 = self.ff1(cond_emb)
        res1 = self.residual_proj1(cond_emb)
        x1 = x1 + res1

        # ff2 + residual
        x2 = self.ff2(x1)
        res2 = self.residual_proj2(x1)
        x2 = x2 + res2

        # ff3 + residual
        x3 = self.ff3(x2)
        res3 = self.residual_proj3(x2)
        x3 = x3 + res3

        # (batch, n*d) -> (batch, n, d)
        x3 = x3.view(x3.shape[0], self.channel_mult[-1], self.in_dim)
        return x3  # (batch, n, d)
    
class Concat_ImageAdapter_3Layers(nn.Module):
    """
    FeedForward 각 블록(ff1, ff2, ff3)마다 residual connection을 적용한 어댑터.
    channel_mult를 3개로 받아서 차원을 점진적으로 확장합니다.
    B,1,1024(clip_emb), B,1,2048(cond_emb) 입력 → 1024로 맞추고 concat → B,2,1024 → flatten → B,2048 → FFN → B,n,2048 출력
    """
    def __init__(self, in_dim=2048, concat_dim=1024, channel_mult=[1, 2, 4], out_dim=2048):
        super().__init__()
        self.in_dim = in_dim
        self.concat_dim = concat_dim
        self.channel_mult = channel_mult
        self.out_dim = out_dim

        # cond_emb: (B, 1, 2048) → (B, 1, 1024)
        self.proj = nn.Linear(in_dim, concat_dim, bias=False)

        # FFN input dim = 2048 (flattened), output 점진적 확장
        flat_dim = concat_dim * 2  # 1024 * 2 = 2048
        dim_out1 = flat_dim * channel_mult[0]
        dim_out2 = flat_dim * channel_mult[1]
        dim_out3 = flat_dim * channel_mult[2]

        self.ff1 = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, dim_out1, bias=False),
            nn.GELU(),
            nn.Linear(dim_out1, dim_out1, bias=False),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim_out1),
            nn.Linear(dim_out1, dim_out2, bias=False),
            nn.GELU(),
            nn.Linear(dim_out2, dim_out2, bias=False),
        )
        self.ff3 = nn.Sequential(
            nn.LayerNorm(dim_out2),
            nn.Linear(dim_out2, dim_out3, bias=False),
            nn.GELU(),
            nn.Linear(dim_out3, dim_out3, bias=False),
        )

        # 각 블록별 residual projection
        self.residual_proj1 = nn.Linear(flat_dim, dim_out1, bias=False) if dim_out1 != flat_dim else nn.Identity()
        self.residual_proj2 = nn.Linear(dim_out1, dim_out2, bias=False) if dim_out2 != dim_out1 else nn.Identity()
        self.residual_proj3 = nn.Linear(dim_out2, dim_out3, bias=False) if dim_out3 != dim_out2 else nn.Identity()

        # 마지막에 원하는 out_dim(2048)로 맞추는 projection
        self.final_proj = nn.Linear(dim_out3, out_dim, bias=False)

    def forward(self, cond_emb, clip_emb):
        # cond_emb: (B, 1, 2048), clip_emb: (B, 1, 1024)
        cond_proj = self.proj(cond_emb)  # (B, 1, 1024)
        concat_emb = torch.cat([clip_emb, cond_proj], dim=1)  # (B, 2, 1024)
        flat_emb = concat_emb.reshape(concat_emb.shape[0], -1)  # (B, 2048)

        x1 = self.ff1(flat_emb)
        res1 = self.residual_proj1(flat_emb)
        x1 = x1 + res1

        x2 = self.ff2(x1)
        res2 = self.residual_proj2(x1)
        x2 = x2 + res2

        x3 = self.ff3(x2)
        res3 = self.residual_proj3(x2)
        x3 = x3 + res3

        # (B, dim_out3) → (B, n, out_dim)
        x3 = x3.view(x3.shape[0], self.channel_mult[-1], self.out_dim)
        return x3  # (B, n, 2048)
    
    
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    
    
class BasicFeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)
        
class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def forward(self, x, context=None, mask=None):

        q = self.to_q(x).unsqueeze(1)  # B*1*(inner_dim)
        k = self.to_k(x).unsqueeze(1)
        v = self.to_v(x).unsqueeze(1)
    
        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out).squeeze(1)  # [B, 1, 1024] -> [B, 1024]


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, out_len = 4, heads=8, dim_head=64, dropout=0):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)

        self.to_out = nn.Sequential( nn.Linear(self.inner_dim, query_dim), nn.Dropout(dropout) )


    def forward(self, x, key, value, mask=None):
        # x: [B, 2048], key/value: [B, 77, 1024]
        
        q = self.to_q(x)     # B*(query_dim)
        q = q.unsqueeze(1)  # B*1*(inner_dim)
        k = key   # B*M*(inner_dim)
        v = value # B*M*(inner_dim)
        print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
        B, Nq, D = q.shape
        print(f"unpack k: {k.shape}, id(k): {id(k)}")
        _, Nk, _ = k.shape
        H = self.heads
        C = self.dim_head

        q = q.view(B, Nq, H, C).transpose(1, 2).reshape(B * H, Nq, C)      # [B*H, Nq, C]
        k = k.view(B, Nk, H, C).transpose(1, 2).reshape(B * H, Nk, C)      # [B*H, Nk, C]
        v = v.view(B, Nk, H, C).transpose(1, 2).reshape(B * H, Nk, C)      # [B*H, Nk, C]

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
        out = (
            out.unsqueeze(0)
            .reshape(B, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out).squeeze(1)  # [B, 1, 1024] -> [B, 1024]
    
class BasicTransformerBlock(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head, use_checkpoint=True):
        super().__init__()
        self.attn1 = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)  
        self.ff = FeedForward(query_dim, glu=True)
        self.attn2 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head)  
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.use_checkpoint = use_checkpoint


    def _forward(self, x, context): 
        x = self.attn1(self.norm1(x))
        print(f"x: {x.shape}")
        x = self.attn1( self.norm1(x) ) + x 
        x = self.attn2(self.norm2(x), context, context) + x
        x = self.ff(self.norm3(x)) + x
        return x
    
    
class BasicTransformerAdapter(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, out_len,  n_heads, d_head):
        super().__init__()
        self.attn1 = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)  
        self.ff = BasicFeedForward(dim = query_dim, dim_out = out_len * query_dim, glu=True)
        self.attn2 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, out_len=out_len, heads=n_heads, dim_head=d_head)  
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.query_dim = query_dim
        self.out_len = out_len


    def forward(self, x, context): 
        x = self.attn1(self.norm1(x)) + x 
        x = self.attn2(self.norm2(x), context, context) + x
        x = self.ff(self.norm3(x))
        x = x.view(x.shape[0], self.out_len, self.query_dim)
        return x