import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, segmentation
from sklearn.decomposition import PCA, TruncatedSVD
import cv2
import os
from glob import glob
import networkx as nx

def extract_and_visualize_eigenvectors(image_path, n_components=5):
    # Load image and convert to grayscale
    img = io.imread(image_path)
    if len(img.shape) == 3:  # Color image
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img
    
    # Vectorize image (2D → 1D)
    height, width = img_gray.shape
    img_vector = img_gray.reshape(-1, 1)  # Column vector for SVD
    
    # Center the data
    img_vector = img_vector - np.mean(img_vector)
    
    # Calculate eigenvectors using SVD
    U, s, Vt = np.linalg.svd(img_vector, full_matrices=False)
    
    # Get top components
    n_comp = min(n_components, len(s))
    eigenvalues = s[:n_comp]**2
    eigenvectors = U[:, :n_comp]
    
    # Visualize main eigenvectors
    plt.figure(figsize=(15, 8))
    
    # Display original image
    plt.subplot(1, n_comp + 1, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Visualize main eigenvectors
    for i in range(n_comp):
        eigenvector = eigenvectors[:, i].reshape(height, width)
        # Normalize for better visualization
        eigenvector = (eigenvector - np.min(eigenvector)) / (np.max(eigenvector) - np.min(eigenvector) + 1e-10)
        plt.subplot(1, n_comp + 1, i + 2)
        plt.imshow(eigenvector, cmap='viridis')
        plt.title(f'Eigenvector {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Visualize eigenvalue contributions
    plt.figure(figsize=(10, 4))
    total = sum(eigenvalues) if sum(eigenvalues) > 0 else 1
    variance_explained = [(i / total) * 100 for i in eigenvalues]
    plt.bar(range(len(variance_explained)), variance_explained)
    plt.ylabel('Explained Variance Ratio (%)')
    plt.xlabel('Principal Components')
    plt.title('Eigenvalue Contributions')
    plt.xticks(range(len(variance_explained)), [f'PC{i+1}' for i in range(len(variance_explained))])
    # Ensure y-axis shows positive values
    plt.ylim(bottom=0)
    plt.show()
    
    return eigenvalues, eigenvectors

def visualize_image_dataset_eigenvectors(image_folder, n_components=10):
    """Visualize the eigenvectors of an image dataset"""
    # Get list of image files
    image_files = glob(os.path.join(image_folder, "*.jpg")) + \
                  glob(os.path.join(image_folder, "*.png"))
    
    if not image_files:
        print("No image files found.")
        return
        
    # Check size using first image
    sample_img = io.imread(image_files[0], as_gray=True)
    height, width = sample_img.shape
    
    # Convert all images to matrix
    image_matrix = []
    for img_path in image_files:
        img = io.imread(img_path, as_gray=True)
        img = cv2.resize(img, (width, height))  # Unify image sizes
        image_matrix.append(img.flatten())
    
    # Create image matrix
    image_matrix = np.array(image_matrix)
    
    # Perform PCA
    actual_components = min(n_components, min(image_matrix.shape))
    pca = PCA(n_components=actual_components)
    pca.fit(image_matrix)
    
    # Visualize eigenvectors
    plt.figure(figsize=(15, 8))
    for i in range(actual_components):
        plt.subplot(2, (actual_components//2) + (actual_components%2), i+1)
        eigenvector = pca.components_[i].reshape(height, width)
        plt.imshow(eigenvector, cmap='viridis')
        plt.title(f'Eigenimage {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Visualize eigenvalues
    plt.figure(figsize=(10, 4))
    plt.bar(range(actual_components), pca.explained_variance_ratio_)
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.title('PCA Eigenvalue Explanation Ratio')
    plt.xticks(range(actual_components), [f'PC{i+1}' for i in range(actual_components)])
    plt.tight_layout()
    plt.show()

def compare_image_eigenvectors(image_paths, n_components=3):
    plt.figure(figsize=(15, 5 * len(image_paths)))
    
    for i, image_path in enumerate(image_paths):
        # Load image and extract eigenvectors
        img = io.imread(image_path)
        if len(img.shape) == 3:
            img_gray = color.rgb2gray(img)
        else:
            img_gray = img
            
        # Use PCA to extract eigenvectors (using sklearn)
        height, width = img_gray.shape
        flattened = img_gray.reshape(1, height * width)
        
        actual_components = min(n_components, min(flattened.shape))
        pca = PCA(n_components=actual_components)
        pca.fit(flattened)
        
        # Display original image
        plt.subplot(len(image_paths), actual_components + 1, i * (actual_components + 1) + 1)
        plt.imshow(img_gray, cmap='gray')
        plt.title(f'Image {i+1}')
        plt.axis('off')
        
        # Visualize main eigenvectors
        for j in range(actual_components):
            component = pca.components_[j].reshape(height, width)
            plt.subplot(len(image_paths), actual_components + 1, i * (actual_components + 1) + j + 2)
            plt.imshow(component, cmap='viridis')
            plt.title(f'Eigenvector {j+1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_all_images_eigenvectors(image_folder, n_components=5, save_results=False, output_folder=None):
    """
    Visualize the eigenvectors of all images in a folder individually.
    
    Parameters:
    -----------
    image_folder : str
        Path to folder containing image files
    n_components : int
        Number of eigenvectors to visualize
    save_results : bool
        Whether to save the results
    output_folder : str
        Folder to save results (required if save_results=True)
    """
    # Get list of image files
    image_files = glob(os.path.join(image_folder, "*.jpg")) + \
                  glob(os.path.join(image_folder, "*.png")) + \
                  glob(os.path.join(image_folder, "*.jpeg"))
    
    if not image_files:
        print("No image files found.")
        return
    
    # Create output folder if saving results
    if save_results:
        if output_folder is None:
            output_folder = os.path.join(image_folder, "results")
        os.makedirs(output_folder, exist_ok=True)
    
    print(f"Found {len(image_files)} image files. Starting processing...")
    
    # Process each image
    for i, img_path in enumerate(image_files):
        print(f"Processing: {i+1}/{len(image_files)} - {os.path.basename(img_path)}")
        
        try:
            # Extract and visualize image eigenvectors
            if save_results:
                # Save mode - don't call plt.show()
                plt.figure(figsize=(15, 8))
                eigenvalues, eigenvectors = extract_and_visualize_eigenvectors_without_show(img_path, n_components)
                
                # Set result file path
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                output_path = os.path.join(output_folder, f"{base_name}_eigenvectors.png")
                
                # Save figure
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create and save eigenvalue graph
                plt.figure(figsize=(10, 4))
                total = sum(eigenvalues) if sum(eigenvalues) > 0 else 1
                variance_explained = [(i / total) * 100 for i in eigenvalues]
                plt.bar(range(len(eigenvalues)), variance_explained)
                plt.ylabel('Explained Variance Ratio (%)')
                plt.xlabel('Principal Components')
                plt.title(f'{os.path.basename(img_path)} - Eigenvalue Contributions')
                plt.xticks(range(len(eigenvalues)), [f'PC{i+1}' for i in range(len(eigenvalues))])
                # Ensure y-axis shows positive values with a reasonable range
                plt.ylim(bottom=0)
                if max(variance_explained) > 0:
                    plt.ylim(top=max(variance_explained) * 1.1)  # Add 10% padding
                
                eigenvalues_path = os.path.join(output_folder, f"{base_name}_eigenvalues.png")
                plt.savefig(eigenvalues_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                # Display mode - show directly
                eigenvalues, eigenvectors = extract_and_visualize_eigenvectors(img_path, n_components)
                plt.suptitle(f'Image: {os.path.basename(img_path)}')
                plt.show()
                
        except Exception as e:
            print(f"Error processing image: {os.path.basename(img_path)} - {str(e)}")
    
    # Integrated eigenvector analysis of all images
    print("Performing integrated eigenvector analysis of all images...")
    try:
        visualize_image_dataset_eigenvectors(image_folder, n_components)
        if save_results:
            output_path = os.path.join(output_folder, "dataset_eigenvectors.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Error during integrated analysis: {str(e)}")

def extract_and_visualize_eigenvectors_without_show(image_path, n_components=5):
    """
    Modified version of extract_and_visualize_eigenvectors that doesn't call plt.show().
    Used for saving results to files.
    """
    # Load image and convert to grayscale
    img = io.imread(image_path)
    if len(img.shape) == 3:  # Color image
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img
    
    # Vectorize image (2D → 1D)
    height, width = img_gray.shape
    img_vector = img_gray.reshape(-1, 1)  # Column vector for SVD
    
    # Center the data
    img_vector = img_vector - np.mean(img_vector)
    
    # Calculate eigenvectors using SVD
    U, s, Vt = np.linalg.svd(img_vector, full_matrices=False)
    
    # Get top components
    n_comp = min(n_components, len(s))
    eigenvalues = s[:n_comp]**2
    eigenvectors = U[:, :n_comp]
    
    # Visualize main eigenvectors (without calling plt.show())
    # Display original image
    plt.subplot(1, n_comp + 1, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Visualize main eigenvectors
    for i in range(n_comp):
        eigenvector = eigenvectors[:, i].reshape(height, width)
        # Normalize for better visualization
        eigenvector = (eigenvector - np.min(eigenvector)) / (np.max(eigenvector) - np.min(eigenvector) + 1e-10)
        plt.subplot(1, n_comp + 1, i + 2)
        plt.imshow(eigenvector, cmap='viridis')
        plt.title(f'Eigenvector {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    
    return eigenvalues, eigenvectors

def visualize_laplacian_similarity(image1_path, image2_path, save_result=False, output_folder=None):
    # 두 이미지 로드
    img1 = io.imread(image1_path)
    img2 = io.imread(image2_path)
    
    # 이미지 크기 통일
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # 세그먼테이션
    segments1 = segmentation.slic(img1, n_segments=30, compactness=10)
    segments2 = segmentation.slic(img2, n_segments=30, compactness=10)
    
    # 두 이미지의 Laplacian 행렬 계산
    L1 = compute_laplacian_matrix(img1, segments1)
    L2 = compute_laplacian_matrix(img2, segments2)
    
    # 두 Laplacian의 고유벡터 추출
    _, eigenvectors1 = np.linalg.eigh(L1)
    _, eigenvectors2 = np.linalg.eigh(L2)
    
    # 고유벡터 유사도 계산 (첫 몇 개의 중요 고유벡터만 사용)
    k = 5  # 사용할 고유벡터 수
    similarity_map = np.zeros_like(segments1, dtype=float)
    
    # 각 세그먼트마다 고유벡터 유사도 계산
    for i in range(np.max(segments1) + 1):
        if i < eigenvectors1.shape[0] and i < eigenvectors2.shape[0]:
            # 해당 세그먼트의 고유벡터 유사도 (코사인 유사도 사용)
            sim = np.abs(np.dot(eigenvectors1[i, :k], eigenvectors2[i, :k]) / 
                         (np.linalg.norm(eigenvectors1[i, :k]) * np.linalg.norm(eigenvectors2[i, :k]) + 1e-10))
            similarity_map[segments1 == i] = sim
    
    # 결과 시각화
    plt.figure(figsize=(15, 5))
    
    # 원본 이미지 2개
    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.title('Image 2')
    plt.axis('off')
    
    # 유사도 맵
    plt.subplot(1, 3, 3)
    plt.imshow(similarity_map, cmap='jet')
    plt.colorbar(label='Similarity')
    plt.title('Laplacian Eigenvector Similarity')
    plt.axis('off')
    
    plt.tight_layout()
    
    # 결과 저장
    if save_result:
        if output_folder is None:
            output_folder = os.path.dirname(image1_path)
        os.makedirs(output_folder, exist_ok=True)
        
        # 파일명 생성
        img1_name = os.path.splitext(os.path.basename(image1_path))[0]
        img2_name = os.path.splitext(os.path.basename(image2_path))[0]
        output_path = os.path.join(output_folder, f"similarity_{img1_name}_vs_{img2_name}.png")
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"유사도 비교 결과 저장됨: {output_path}")
    
    plt.show()
    
    return similarity_map

def compare_eigenvectors_visually(image1_path, image2_path, n_eigenvectors=3, save_result=False, output_folder=None):
    # 두 이미지의 고유벡터 계산
    img1 = io.imread(image1_path)
    img2 = io.imread(image2_path)
    
    segments1, eigenvalues1, eigenvectors1 = compute_image_eigenvectors(img1)
    segments2, eigenvalues2, eigenvectors2 = compute_image_eigenvectors(img2)
    
    # 결과 시각화 - 레이아웃을 수정하여 인덱스 오류 방지
    # nrows를 n_eigenvectors + 1로 설정하여 원본 이미지와 각 고유벡터를 별도 행에 배치
    plt.figure(figsize=(15, 5 * (n_eigenvectors + 1)))
    
    # 원본 이미지 표시 (첫 번째 행)
    plt.subplot(n_eigenvectors + 1, 3, 1)
    plt.imshow(img1)
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(n_eigenvectors + 1, 3, 2)
    plt.imshow(img2)
    plt.title('Image 2')
    plt.axis('off')
    
    plt.subplot(n_eigenvectors + 1, 3, 3)
    plt.axis('off')  # 빈 칸
    
    # 각 고유벡터를 시각화하고 차이 표시 (2번째 행부터)
    for i in range(n_eigenvectors):
        # 첫 번째 이미지의 고유벡터
        plt.subplot(n_eigenvectors + 1, 3, (i + 1) * 3 + 1)
        eigen_map1 = visualize_eigenvector(eigenvectors1[:, i], segments1)
        plt.imshow(eigen_map1, cmap='viridis')
        plt.title(f'Image 1: Eigenvector {i+1}')
        plt.axis('off')
        
        # 두 번째 이미지의 고유벡터
        plt.subplot(n_eigenvectors + 1, 3, (i + 1) * 3 + 2)
        eigen_map2 = visualize_eigenvector(eigenvectors2[:, i], segments2)
        plt.imshow(eigen_map2, cmap='viridis')
        plt.title(f'Image 2: Eigenvector {i+1}')
        plt.axis('off')
        
        # 두 고유벡터의 차이 시각화
        plt.subplot(n_eigenvectors + 1, 3, (i + 1) * 3 + 3)
        # 크기가 다르면 리사이즈
        if eigen_map1.shape != eigen_map2.shape:
            eigen_map2_resized = cv2.resize(eigen_map2, (eigen_map1.shape[1], eigen_map1.shape[0]))
            diff = np.abs(eigen_map1 - eigen_map2_resized)
        else:
            diff = np.abs(eigen_map1 - eigen_map2)
            
        plt.imshow(diff, cmap='hot')
        plt.title(f'Difference: Eigenvector {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    
    # 결과 저장
    if save_result:
        if output_folder is None:
            output_folder = os.path.dirname(image1_path)
        os.makedirs(output_folder, exist_ok=True)
        
        # 파일명 생성
        img1_name = os.path.splitext(os.path.basename(image1_path))[0]
        img2_name = os.path.splitext(os.path.basename(image2_path))[0]
        output_path = os.path.join(output_folder, f"eigenvector_comparison_{img1_name}_vs_{img2_name}.png")
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"고유벡터 비교 결과 저장됨: {output_path}")
    
    plt.show()

def compute_laplacian_matrix(img, segments):
    """
    이미지와 세그먼트 정보로부터 Laplacian 행렬을 계산합니다.
    NetworkX의 normalized_laplacian_matrix 사용 대신 직접 구현
    """
    # 그래프 생성
    G = nx.Graph()
    
    # 노드 추가 (각 세그먼트)
    for i in range(np.max(segments) + 1):
        # 세그먼트의 평균 색상 계산
        mask = segments == i
        if np.any(mask):
            if len(img.shape) == 3:  # 컬러 이미지
                avg_color = np.mean(img[mask], axis=0)
            else:  # 그레이스케일 이미지
                avg_color = np.mean(img[mask])
            G.add_node(i, color=avg_color)
    
    # 엣지 추가 (인접한 세그먼트 간)
    height, width = segments.shape[:2]
    for i in range(height-1):
        for j in range(width-1):
            current = segments[i, j]
            right = segments[i, j+1]
            below = segments[i+1, j]
            
            if current != right and not G.has_edge(current, right):
                # 색상 유사도 계산
                if hasattr(G.nodes[current], 'color') and hasattr(G.nodes[right], 'color'):
                    color_diff = np.sum((G.nodes[current]['color'] - G.nodes[right]['color'])**2)
                    weight = np.exp(-color_diff/1000)  # 색상 차이를 가중치로 변환
                    G.add_edge(current, right, weight=weight)
            
            if current != below and not G.has_edge(current, below):
                # 색상 유사도 계산
                if hasattr(G.nodes[current], 'color') and hasattr(G.nodes[below], 'color'):
                    color_diff = np.sum((G.nodes[current]['color'] - G.nodes[below]['color'])**2)
                    weight = np.exp(-color_diff/1000)
                    G.add_edge(current, below, weight=weight)
    
    # 직접 Normalized Laplacian 행렬 계산 (NetworkX의 함수 대신 사용)
    # 1. 인접 행렬 만들기
    n = len(G.nodes)
    A = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0)
        A[u, v] = weight
        A[v, u] = weight  # 무방향 그래프
    
    # 2. 차수 행렬 만들기
    D = np.zeros((n, n))
    for i in range(n):
        degree = np.sum(A[i, :])
        if degree > 0:
            D[i, i] = degree
    
    # 3. 정규화된 Laplacian 행렬 계산
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                if D[i, i] > 0:
                    L[i, j] = 1.0
            else:
                if D[i, i] > 0 and D[j, j] > 0 and A[i, j] > 0:
                    L[i, j] = -A[i, j] / np.sqrt(D[i, i] * D[j, j])
    
    return L

def compute_image_eigenvectors(img, n_segments=50, n_components=5):
    """
    이미지를 세그먼트로 나누고 Laplacian 행렬의 고유벡터를 계산합니다.
    """
    # 세그먼테이션
    segments = segmentation.slic(img, n_segments=n_segments, compactness=10)
    
    # Laplacian 행렬 계산
    L = compute_laplacian_matrix(img, segments)
    
    # 고유값과 고유벡터 계산
    if L.shape[0] > 0:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # 작은 고유값 순으로 정렬 (중요도 순)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 주요 컴포넌트만 선택
        n_comp = min(n_components, len(eigenvalues))
        eigenvalues = eigenvalues[:n_comp]
        eigenvectors = eigenvectors[:, :n_comp]
    else:
        # 예외 처리: 세그먼트가 없는 경우
        eigenvalues = np.array([])
        eigenvectors = np.array([]).reshape(0, 0)
    
    return segments, eigenvalues, eigenvectors

def visualize_eigenvector(eigenvector, segments):
    """
    고유벡터를 시각화합니다.
    """
    # 고유벡터 값을 세그먼트에 매핑
    eigen_map = np.zeros_like(segments, dtype=float)
    for i in range(min(len(eigenvector), np.max(segments) + 1)):
        eigen_map[segments == i] = eigenvector[i]
    
    # 정규화
    eigen_map = (eigen_map - np.min(eigen_map)) / (np.max(eigen_map) - np.min(eigen_map) + 1e-10)
    
    return eigen_map

def compare_laplacian_matrices(image1_path, image2_path, save_result=False, output_folder=None):
    """두 이미지의 Laplacian 행렬을 직접 비교하고 시각화합니다."""
    # 이미지 로드
    img1 = io.imread(image1_path)
    img2 = io.imread(image2_path)
    
    # 세그먼테이션
    segments1 = segmentation.slic(img1, n_segments=50, compactness=10)
    segments2 = segmentation.slic(img2, n_segments=50, compactness=10)
    
    # Laplacian 행렬 계산
    L1 = compute_laplacian_matrix(img1, segments1)
    L2 = compute_laplacian_matrix(img2, segments2)
    
    # 행렬 크기 맞추기 (더 작은 쪽에 맞춤)
    min_size = min(L1.shape[0], L2.shape[0])
    L1_resized = L1[:min_size, :min_size]
    L2_resized = L2[:min_size, :min_size]
    
    # 결과 시각화
    plt.figure(figsize=(18, 6))
    
    # 원본 이미지 2개
    plt.subplot(2, 3, 1)
    plt.imshow(img1)
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(img2)
    plt.title('Image 2')
    plt.axis('off')
    
    # Laplacian 행렬 1 시각화
    plt.subplot(2, 3, 2)
    im1 = plt.imshow(L1_resized, cmap='viridis')
    plt.colorbar(im1)
    plt.title('Laplacian Matrix - Image 1')
    
    # Laplacian 행렬 2 시각화
    plt.subplot(2, 3, 5)
    im2 = plt.imshow(L2_resized, cmap='viridis')
    plt.colorbar(im2)
    plt.title('Laplacian Matrix - Image 2')
    
    # 두 행렬의 차이 계산 및 시각화
    diff = np.abs(L1_resized - L2_resized)
    
    plt.subplot(2, 3, 3)
    im_diff = plt.imshow(diff, cmap='hot')
    plt.colorbar(im_diff)
    plt.title('Absolute Difference')
    
    # Frobenius norm으로 전체 차이 정량화
    frob_norm = np.linalg.norm(diff, 'fro')
    plt.figtext(0.5, 0.01, f'Frobenius Norm of Difference: {frob_norm:.4f}', ha='center')
    
    # 행렬 구조 유사도 (코사인 유사도)
    L1_flat = L1_resized.flatten()
    L2_flat = L2_resized.flatten()
    cos_sim = np.dot(L1_flat, L2_flat) / (np.linalg.norm(L1_flat) * np.linalg.norm(L2_flat) + 1e-10)
    plt.figtext(0.5, 0.04, f'Cosine Similarity: {cos_sim:.4f}', ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # 결과 저장
    if save_result:
        if output_folder is None:
            output_folder = os.path.dirname(image1_path)
        os.makedirs(output_folder, exist_ok=True)
        
        # 파일명 생성
        img1_name = os.path.splitext(os.path.basename(image1_path))[0]
        img2_name = os.path.splitext(os.path.basename(image2_path))[0]
        output_path = os.path.join(output_folder, f"laplacian_matrix_comparison_{img1_name}_vs_{img2_name}.png")
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Laplacian 행렬 비교 결과 저장됨: {output_path}")
    
    plt.show()
    
    return L1_resized, L2_resized, diff, frob_norm, cos_sim

def compare_svd_features(image_folder, n_components=5):
    # 폴더에서 이미지 파일 가져오기
    image_files = glob(os.path.join(image_folder, "*.jpg")) + \
                  glob(os.path.join(image_folder, "*.png")) + \
                  glob(os.path.join(image_folder, "*.jpeg"))
    
    if len(image_files) < 2:
        print("최소 2개 이상의 이미지가 필요합니다.")
        return
    
    # 처음 두 개의 이미지만 선택
    image_path1 = image_files[0]
    image_path2 = image_files[1]
    
    # 첫 번째 이미지 처리
    img1 = io.imread(image_path1)
    if len(img1.shape) == 3:  # 컬러 이미지
        img1_gray = color.rgb2gray(img1)
    else:
        img1_gray = img1
    
    # 두 번째 이미지 처리
    img2 = io.imread(image_path2)
    if len(img2.shape) == 3:  # 컬러 이미지
        img2_gray = color.rgb2gray(img2)
    else:
        img2_gray = img2
    
    # 이미지 1의 SVD 적용
    height1, width1 = img1_gray.shape
    img1_vector = img1_gray.reshape(-1, 1)
    img1_vector = img1_vector - np.mean(img1_vector)
    U1, s1, Vt1 = np.linalg.svd(img1_vector, full_matrices=False)
    
    # 이미지 2의 SVD 적용
    height2, width2 = img2_gray.shape
    img2_vector = img2_gray.reshape(-1, 1)
    img2_vector = img2_vector - np.mean(img2_vector)
    U2, s2, Vt2 = np.linalg.svd(img2_vector, full_matrices=False)
    
    # Shape 정보 출력
    print("===== SVD 적용 후 shape 정보 =====")
    print(f"이미지 1 원본 shape: {img1_gray.shape}")
    print(f"이미지 1 벡터화 shape: {img1_vector.shape}")
    print(f"이미지 1 U 행렬 shape: {U1.shape}")
    print(f"이미지 1 특이값(s) shape: {s1.shape}")
    print(f"이미지 1 Vt 행렬 shape: {Vt1.shape}")
    print()
    print(f"이미지 2 원본 shape: {img2_gray.shape}")
    print(f"이미지 2 벡터화 shape: {img2_vector.shape}")
    print(f"이미지 2 U 행렬 shape: {U2.shape}")
    print(f"이미지 2 특이값(s) shape: {s2.shape}")
    print(f"이미지 2 Vt 행렬 shape: {Vt2.shape}")
    
    # 주요 특징 성분 가져오기
    n_comp = min(n_components, len(s1), len(s2))
    eigenvalues1 = s1[:n_comp]**2
    eigenvectors1 = U1[:, :n_comp]
    eigenvalues2 = s2[:n_comp]**2
    eigenvectors2 = U2[:, :n_comp]
    
    print("\n===== 고유벡터 shape 정보 =====")
    print(f"이미지 1 고유값 shape: {eigenvalues1.shape}")
    print(f"이미지 1 고유벡터 shape: {eigenvectors1.shape}")
    print(f"이미지 2 고유값 shape: {eigenvalues2.shape}")
    print(f"이미지 2 고유벡터 shape: {eigenvectors2.shape}")
    
    # 두 이미지의 고유벡터 시각화 및 비교
    plt.figure(figsize=(15, 4 * n_comp))
    
    # 원본 이미지 표시
    plt.subplot(n_comp + 1, 3, 1)
    plt.imshow(img1_gray, cmap='gray')
    plt.title(f'이미지 1: {os.path.basename(image_path1)}')
    plt.axis('off')
    
    plt.subplot(n_comp + 1, 3, 2)
    plt.imshow(img2_gray, cmap='gray')
    plt.title(f'이미지 2: {os.path.basename(image_path2)}')
    plt.axis('off')
    
    plt.subplot(n_comp + 1, 3, 3)
    plt.axis('off')  # 빈 공간
    
    # 고유벡터 시각화 및 비교
    for i in range(n_comp):
        # 이미지 1의 고유벡터
        eigenvector1 = eigenvectors1[:, i].reshape(height1, width1)
        # 정규화
        eigenvector1 = (eigenvector1 - np.min(eigenvector1)) / (np.max(eigenvector1) - np.min(eigenvector1) + 1e-10)
        
        plt.subplot(n_comp + 1, 3, (i + 1) * 3 + 1)
        plt.imshow(eigenvector1, cmap='viridis')
        plt.title(f'이미지 1: 고유벡터 {i+1}')
        plt.axis('off')
        
        # 이미지 2의 고유벡터
        eigenvector2 = eigenvectors2[:, i].reshape(height2, width2)
        # 정규화
        eigenvector2 = (eigenvector2 - np.min(eigenvector2)) / (np.max(eigenvector2) - np.min(eigenvector2) + 1e-10)
        
        plt.subplot(n_comp + 1, 3, (i + 1) * 3 + 2)
        plt.imshow(eigenvector2, cmap='viridis')
        plt.title(f'이미지 2: 고유벡터 {i+1}')
        plt.axis('off')
        
        # 고유값 기여도 비교 (막대 그래프)
        plt.subplot(n_comp + 1, 3, (i + 1) * 3 + 3)
        contribution1 = eigenvalues1[i] / np.sum(eigenvalues1) * 100
        contribution2 = eigenvalues2[i] / np.sum(eigenvalues2) * 100
        plt.bar(['이미지 1', '이미지 2'], [contribution1, contribution2])
        plt.title(f'고유벡터 {i+1} 기여도 비교 (%)')
        plt.ylim(0, max(contribution1, contribution2) * 1.2)
    
    plt.tight_layout()
    plt.show()
    
    return eigenvectors1, eigenvectors2, eigenvalues1, eigenvalues2

# Main code: Use this part to call functions
if __name__ == "__main__":
    # Specify image folder path
    image_folder = "/workspace/data/changhyun/projects/emoji_generation/Controlnet_custom/eigen"
    
    # 결과 저장 디렉토리 설정
    save_outputs = True  # 결과 저장 활성화
    results_path = os.path.join(image_folder, "results")
    if save_outputs:
        # Clear previous results directory if it exists
        import shutil
        if os.path.exists(results_path):
            shutil.rmtree(results_path)
        os.makedirs(results_path, exist_ok=True)
        print(f"결과가 저장될 디렉토리: {results_path}")
    
    # 이미지 파일 목록 가져오기
    image_files = glob(os.path.join(image_folder, "*.jpg")) + \
                  glob(os.path.join(image_folder, "*.png")) + \
                  glob(os.path.join(image_folder, "*.jpeg"))
    
    print(f"Found {len(image_files)} images in directory: {image_folder}")
    for i, f in enumerate(image_files):
        print(f"  {i+1}. {os.path.basename(f)}")
    
    # 두 이미지 비교 부분만 독립적으로 실행
    if len(image_files) >= 2:
        print("\n======= 이미지 비교 시작 =======")
        
        try:
            print("1. Laplacian 행렬 직접 비교 실행...")
            compare_laplacian_matrices(image_files[0], image_files[1], save_result=save_outputs, output_folder=results_path)
            print("✓ Laplacian 행렬 비교 완료")
        except Exception as e:
            print(f"✗ Laplacian 행렬 비교 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
        
        try:
            print("\n2. Laplacian 유사도 비교 실행...")
            visualize_laplacian_similarity(image_files[0], image_files[1], save_result=save_outputs, output_folder=results_path)
            print("✓ Laplacian 유사도 비교 완료")
        except Exception as e:
            print(f"✗ Laplacian 유사도 비교 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
        
        try:
            print("\n3. 고유벡터 시각적 비교 실행...")
            compare_eigenvectors_visually(image_files[0], image_files[1], n_eigenvectors=20, 
                                        save_result=save_outputs, output_folder=results_path)  # 고유벡터 수를 2개로 줄임
            print("✓ 고유벡터 시각적 비교 완료")
        except Exception as e:
            print(f"✗ 고유벡터 시각적 비교 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("\n비교할 이미지가 충분하지 않습니다. 최소 2개 이상의 이미지가 필요합니다.")
    
    # 모든 이미지의 고유벡터 시각화 옵션
    if save_outputs:
        try:
            print("\n4. 모든 이미지의 고유벡터 시각화...")
            visualize_all_images_eigenvectors(image_folder, n_components=3, save_results=True, 
                                             output_folder=results_path)
            print("✓ 모든 이미지의 고유벡터 시각화 완료")
        except Exception as e:
            print(f"✗ 모든 이미지의 고유벡터 시각화 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
