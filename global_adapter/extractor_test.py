import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from data import Data
from network import MGN
from opt import opt
from PIL import Image

def run_tsne_and_save_grids():
    extractor = MGN().to('cuda')
    extractor.eval()
    extractor.load_state_dict(torch.load(opt.extractor_weight, map_location='cuda'))

    data = Data()
    test_loader = data.test_loader

    features = []
    labels = []
    filenames = []
    images_raw = []
    count = 0
    max_samples = 100

    with torch.no_grad():
        for imgs, lbls, captions in test_loader:
            imgs = imgs.to('cuda')
            outs = extractor(imgs)
            feat = outs[0].detach().cpu().numpy()
            features.append(feat)
            labels.append(lbls.numpy())
            # 파일명 대신 인덱스 사용
            filenames.extend([f"sample_{count+i}" for i in range(feat.shape[0])])
            # 원본 이미지 저장 (PIL 변환)
            for img in imgs.cpu():
                img = (img + 1) / 2  # [-1,1] -> [0,1]
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).clip(0, 255).astype(np.uint8)
                images_raw.append(Image.fromarray(img))
            count += feat.shape[0]
            if count >= max_samples:
                break

    features = np.concatenate(features, axis=0)[:max_samples]
    labels = np.concatenate(labels, axis=0)[:max_samples]
    filenames = filenames[:max_samples]
    images_raw = images_raw[:max_samples]

    # t-SNE 적용
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # 각 샘플별로 거리 계산 (유클리드 거리)
    from scipy.spatial.distance import cdist
    dists = cdist(features_2d, features_2d, metric='euclidean')

    # 각 샘플별로 가까운 이웃 5개 인덱스(자기 자신 포함)
    nearest_indices = np.argsort(dists, axis=1)[:, :5]

    # "가까운 이웃들의 평균 거리"가 가장 작은 5개 샘플 선택 (즉, 밀집된 샘플)
    mean_nearest_dist = np.mean(np.take_along_axis(dists, nearest_indices, axis=1), axis=1)
    top5_dense_indices = np.argsort(mean_nearest_dist)[:5]

    for idx, center_idx in enumerate(top5_dense_indices):
        neighbor_idxs = nearest_indices[center_idx]
        grid_imgs = [images_raw[i] for i in neighbor_idxs]
        # 그리드로 저장 (1행 5열)
        grid_w, grid_h = grid_imgs[0].width * 5, grid_imgs[0].height
        grid = Image.new('RGB', (grid_w, grid_h))
        for i, img in enumerate(grid_imgs):
            grid.paste(img, (i * img.width, 0))
        grid.save(f"tsne_dense_sample_{idx+1}.png")
        print(f"tsne_dense_sample_{idx+1}.png 저장 완료 (중심: {filenames[center_idx]})")

if __name__ == "__main__":
    run_tsne_and_save_grids()
