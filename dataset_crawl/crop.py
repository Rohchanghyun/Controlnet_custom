import cv2
import numpy as np
from PIL import Image
import os
import json
import torch
from segment_anything import sam_model_registry, SamPredictor
from scipy.cluster.hierarchy import fcluster, linkage

def find_component_centers(img, min_alpha=50):
    """이미지에서 포인트를 더 조밀하게 찾기"""
    alpha = img[:, :, 3]
    binary = (alpha > min_alpha).astype(np.uint8)
    
    # 연결된 컴포넌트 찾기
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    # 각 컴포넌트에 대해 여러 포인트 생성
    points = []
    for i in range(1, num_labels):  # 0은 배경이므로 제외
        x, y, w, h, area = stats[i]
        if area < 1000:  # 너무 작은 영역 무시
            continue
            
        # 컴포넌트 영역 내에서 격자 형태로 포인트 생성
        step = 20  # 격자 간격 (작을수록 더 많은 포인트)
        for row in range(y, y + h, step):
            for col in range(x, x + w, step):
                if row < binary.shape[0] and col < binary.shape[1]:
                    if binary[row, col] > 0:  # 알파 ��이 있는 부분만
                        points.append([col, row])
        
        # 컴포넌트의 중심점도 추가
        cx = x + w//2
        cy = y + h//2
        points.append([cx, cy])
        
        # 경계 상자의 코너 포인트들도 추가
        points.extend([
            [x, y],           # 좌상
            [x + w, y],       # 우상
            [x, y + h],       # 좌하
            [x + w, y + h]    # 우하
        ])
    
    return np.array(points) if points else np.array([])

def post_process_mask(mask):
    # 이진 마스크로 변환
    binary_mask = mask.astype(np.uint8)
    
    # 커널 크기 정의 (필요에 따라 조정 가능)
    kernel = np.ones((5,5), np.uint8)
    
    # 닫기 연산 (작은 구멍 메우기)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # 열기 연산 (노이즈 제거)
    final_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    
    return final_mask > 0

def group_stickers_by_location(stats):
    # stats에서 각 스티커의 중심점 추출
    centers = []
    valid_indices = []
    
    for i in range(1, len(stats)):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 100:  # 너무 작은 영역 무시
            continue
            
        x = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] // 2
        y = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] // 2
        centers.append([x, y])
        valid_indices.append(i)
    
    if len(centers) < 2:
        return np.zeros(len(stats) - 1)
    
    centers = np.array(centers)
    
    # 계층적 클러스터링 수행
    linkage_matrix = linkage(centers, method='ward')
    
    # 거리 임계값으로 클러스터 분 (이 값을 조정하여 그룹화 정도 제어)
    distance_threshold = 100  # 픽셀 단위
    clusters = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    
    # 모든 컴포넌트에 대한 그룹 할당
    final_groups = np.zeros(len(stats) - 1)
    for idx, valid_idx in enumerate(valid_indices):
        final_groups[valid_idx - 1] = clusters[idx] - 1
    
    return final_groups

def extract_stickers_sam(image_path, output_dir, stats_dict, sam_predictor):
    print(f"\nProcessing image: {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return
    
    if img.shape[2] == 4:
        rgb_img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(rgb_img)
        
        input_points = find_component_centers(img)
        if len(input_points) == 0:
            print(f"No points found in {image_path}")
            return
            
        input_labels = np.ones(len(input_points))
        
        masks, scores, logits = sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        # 가장 높은 점수의 마스크 찾기
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        
        # 최고 점수 마스크 후처리
        processed_mask = post_process_mask(best_mask)
        mask_image = (processed_mask * 255).astype(np.uint8)
        
        # 연결된 컴포넌트 찾기
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_image)
        
        # 각 연결 컴포넌트에 대해 개별 이미지 저장
        for label_idx in range(1, num_labels):  # 0은 배경이므로 제외
            # 현재 라벨에 대한 마스크 생성
            component_mask = (labels == label_idx)
            
            # 컴포넌트의 경계 상자 정보
            x = stats[label_idx, cv2.CC_STAT_LEFT]
            y = stats[label_idx, cv2.CC_STAT_TOP]
            w = stats[label_idx, cv2.CC_STAT_WIDTH]
            h = stats[label_idx, cv2.CC_STAT_HEIGHT]
            area = stats[label_idx, cv2.CC_STAT_AREA]
            
            # 너버깅 정보 출력
            print(f"\nComponent {label_idx} info:")
            print(f"Width x Height: {w} x {h}")
            print(f"Bounding box area: {w * h}")
            print(f"Actual pixel area: {area}")
            
            # 너무 작은 컴포넌트는 건너뛰기
            if area < 1000:  # 최소 크기 임계값
                print(f"Skipping component {label_idx} due to small area")
                continue
            
            # 컴포넌트 마스크로 이미지 추출
            component_img = img.copy()
            # 배경을 흰색으로 설정 (RGB: 255, 255, 255, Alpha: 255)
            white_background = np.ones_like(component_img) * 255
            white_background[:, :, 3] = 255  # 알파 채널을 255로 설정 (불투명)
            
            # 마스크를 사용하여 원본 이미지와 흰색 배경 합성
            component_img = np.where(component_mask[:, :, np.newaxis], component_img, white_background)
            
            # 경계 상자로 이미지 자르기
            cropped_img = component_img[y:y+h, x:x+w]
            
            # BGR을 RGB로 변환 (알파 채널 유지)
            cropped_img_rgb = np.zeros_like(cropped_img)
            cropped_img_rgb[:, :, 0:3] = cv2.cvtColor(cropped_img[:, :, 0:3], cv2.COLOR_BGR2RGB)
            cropped_img_rgb[:, :, 3] = cropped_img[:, :, 3]
            
            # 잘린 이미지 저장
            try:
                cropped_pil = Image.fromarray(cropped_img_rgb)
                output_path = os.path.join(output_dir, 
                    f"{os.path.splitext(os.path.basename(image_path))[0]}_component_{label_idx}_score_{best_score:.3f}_area_{area}_size_{w}x{h}.png")
                cropped_pil.save(output_path, "PNG")
                print(f"Saved component {label_idx}: {output_path}")
            except Exception as e:
                print(f"Failed to save component {label_idx}: {e}")

# SAM 모델 초기화
def initialize_sam():
    print("Initializing SAM model...")
    sam = sam_model_registry["vit_h"](checkpoint="./SAM_weight/sam_vit_h_4b8939.pth")
    sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    return SamPredictor(sam)

# 메인 실행
if __name__ == "__main__":
    # 디렉토리 정
    input_dir = "/workspace/dataset/character_emoticon_data/emoji_crawl"
    output_dir = "/workspace/dataset/character_emoticon_data/cropped_sticker"
    os.makedirs(output_dir, exist_ok=True)
    
    # SAM 초기화
    sam_predictor = initialize_sam()
    
    # 통계 정보를 저장할 딕셔너리
    stats_dict = {}
    
    # 처리 파일 수 카운트
    processed_files = 0
    successful_crops = 0
    failed_crops = 0
    
    # 모든 이미지 처리
    print("\nStarting image processing...")
    for filename in os.listdir(input_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            processed_files += 1
            image_path = os.path.join(input_dir, filename)
            try:
                extract_stickers_sam(image_path, output_dir, stats_dict, sam_predictor)
                successful_crops += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                failed_crops += 1
    
    # JSON 파일로 저장
    json_path = os.path.join(output_dir, "sticker_stats.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=4)
    
    print("\nProcessing Summary:")
    print(f"Total files processed: {processed_files}")
    print(f"Successful crops: {successful_crops}")
    print(f"Failed crops: {failed_crops}")
    print(f"Statistics saved to {json_path}")
