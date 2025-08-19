import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck
from diffusers.models.attention_processor import Attention

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
    def __init__(self, extractor_dim=2048, hidden_dim=1024, clip_embeddings_dim=512):
        super(Image_adapter, self).__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(extractor_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, clip_embeddings_dim),
            torch.nn.LayerNorm(clip_embeddings_dim)
        )
        
    #     self._init_weights()

    # def _init_weights(self):
    #     for m in self.proj:
    #         if isinstance(m, nn.Linear):
    #             nn.init.constant_(m.weight, 0)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 0)

    def forward(self, image_embeds):
        projection_embedding = self.proj(image_embeds)
        return projection_embedding 
    
class Image_adapter_77(nn.Module):
    """project image embedding to CLIP embedding space"""
    def __init__(self, extractor_dim=2048, hidden_dim=1024, clip_embeddings_dim=77 * 512):
        super(Image_adapter_77, self).__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(extractor_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, clip_embeddings_dim),
            torch.nn.LayerNorm(clip_embeddings_dim)
        )
        
    def forward(self, image_embeds):
        projection_embedding = self.proj(image_embeds)
        projection_embedding = projection_embedding.view(-1, 77, 512)
        return projection_embedding 

class Image_adapter_77_768(nn.Module):
    """project image embedding to CLIP embedding space"""
    def __init__(self, extractor_dim=2048, hidden_dim=1024, clip_embeddings_dim=77 * 768):
        super(Image_adapter_77_768, self).__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(extractor_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, clip_embeddings_dim),
            torch.nn.LayerNorm(clip_embeddings_dim)
        )
        
    def forward(self, image_embeds):
        projection_embedding = self.proj(image_embeds)
        projection_embedding = projection_embedding.view(-1, 77, 768)
        return projection_embedding 

class Image_adapter_4_768(nn.Module):
    """project image embedding to CLIP embedding space"""
    def __init__(self, extractor_dim=2048, hidden_dim=1024, clip_embeddings_dim=77 * 768):
        super(Image_adapter_4_768, self).__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(extractor_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, clip_embeddings_dim),
            torch.nn.LayerNorm(clip_embeddings_dim)
        )
        
    def forward(self, image_embeds):
        projection_embedding = self.proj(image_embeds)
        projection_embedding = projection_embedding.view(-1, 77, 768)
        return projection_embedding 

class Image_adapter_768(nn.Module):
    """project image embedding to CLIP embedding space"""
    def __init__(self, extractor_dim=2048, hidden_dim=1024, clip_embeddings_dim=768):
        super(Image_adapter_768, self).__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(extractor_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, clip_embeddings_dim),
            torch.nn.LayerNorm(clip_embeddings_dim)
        )
        
    #     self._init_weights()

    # def _init_weights(self):
    #     for m in self.proj:
    #         if isinstance(m, nn.Linear):
    #             nn.init.constant_(m.weight, 0)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 0)

    def forward(self, image_embeds):
        projection_embedding = image_embeds
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

class ImageTokenAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cnn = nn.Sequential(
            # (3, 512, 512) -> (32, 256, 256)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # (32, 256, 256) -> (64, 128, 128)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # (64, 128, 128) -> (128, 64, 64)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # (128, 64, 64) -> (256, 32, 32)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 두 개의 projection head
        self.proj_clip = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),     # (256, 8, 8) -> 16384
            nn.Flatten(),                      # 16384
            nn.Linear(256 * 8 * 8, 2048),     # 16384 -> 2048 (근본있게)
            nn.ReLU(inplace=True),
            nn.Linear(2048, 768 * 4),         # 2048 -> (4, 768) for CLIP v1
        )
        
        self.proj_laion = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),     # (256, 8, 8) -> 16384
            nn.Flatten(),                      # 16384
            nn.Linear(256 * 8 * 8, 2048),     # 16384 -> 2048 (근본있게)
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1280 * 4),        # 2048 -> (4, 1280) for LAION CLIP
        )

    def forward(self, x):
        x = self.cnn(x)
        clip_tokens = self.proj_clip(x).view(-1, 4, 768)    # [B, 4, 768]
        laion_tokens = self.proj_laion(x).view(-1, 4, 1280) # [B, 4, 1280]
        return clip_tokens, laion_tokens

class VisualTokenProjector(nn.Module):
    """Project visual tokens into CLIP and LAION CLIP embedding spaces"""
    def __init__(self, input_dim=768):
        super().__init__()
        
        # CLIP projection (768 -> 768)
        self.clip_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 768),
            nn.LayerNorm(768)
        )
        
        # LAION CLIP projection (768 -> 1280)
        self.laion_proj = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.GELU(),
            nn.Linear(768, 1280),
            nn.LayerNorm(1280)
        )

    def forward(self, visual_tokens):
        # visual_tokens: [B, 768, 192]
        # Transpose to [B, 192, 768] for easier processing
        visual_tokens = visual_tokens[:, :768, :]
        visual_tokens = visual_tokens.transpose(1, 2)  # [B, 192, 768]
        
        # Project to each embedding space
        clip_embeddings = self.clip_proj(visual_tokens)    # [B, 192, 768]
        laion_embeddings = self.laion_proj(visual_tokens)  # [B, 192, 1280]
        
        # 이제 transpose 하지 않고 바로 반환
        return clip_embeddings, laion_embeddings  # [B, 192, 768], [B, 192, 1280]

class TokenImageAdapter(nn.Module):
    """Adapter for combining visual tokens and sketch images"""
    def __init__(self, token_dim=768, hidden_dim=1024):
        super().__init__()
        
        # Projection layer for concatenated features
        self.proj = nn.Sequential(
            nn.Linear(960, hidden_dim),  # 960 = 192(token) + 768(image)
            nn.GELU(),
            nn.Linear(hidden_dim, 768),
            nn.LayerNorm(768)
        )
        
    def forward(self, visual_tokens, sketch_images):
        """
        Args:
            visual_tokens: (B, 192, 768)
            sketch_images: (B, 3, 768, 768)
        Returns:
            output: (B, 3, 768, 768)
        """
        B = visual_tokens.shape[0]
        
        # 1. Duplicate visual tokens for each channel
        visual_tokens = visual_tokens.unsqueeze(1)  # (B, 1, 192, 768)
        visual_tokens = visual_tokens.permute(0, 1, 3, 2)
        visual_tokens = visual_tokens.expand(-1, 3, -1, -1)  # (B, 3, 192, 768)
        
        #print("visual_tokens", visual_tokens.shape) # visual_tokens torch.Size([1, 3, 192, 785])
        #print("sketch_images", sketch_images.shape) # sketch_images torch.Size([1, 3, 768, 768])

        visual_tokens = visual_tokens[:, :, :, :768]
        # 2. Concatenate along the spatial dimension
        #print("visual_tokens after slicing", visual_tokens.shape) # visual_tokens torch.Size([1, 3, 192, 768])
        combined = torch.cat([visual_tokens, sketch_images], dim=2)  # (B, 3, 960, 768)
        
        # 3. Project back to original spatial dimensions
        # Reshape for linear projection while maintaining H,W order
        combined = combined.permute(0, 1, 3, 2)  # (B, 3, 768, 960)
        output = self.proj(combined)  # (B, 3, 768, 768)
        output = output.permute(0, 1, 3, 2)  # (B, 3, 768, 768) - H,W 순서 복원
        
        return output