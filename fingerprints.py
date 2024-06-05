import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def get_fp_feature(img, flg_show):
    # 전처리: CLAHE를 사용한 이미지 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # 전처리: 가우시안 블러 적용
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # 전처리: 적응형 이진화
    binary_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 전처리: 모폴로지 닫힘 연산 적용 (노이즈 제거)
    kernel = np.ones((3, 3), np.uint8)
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # 전처리: 모폴로지 팽창 연산 적용 (선을 두껍게 만듦)
    dilated_img = cv2.dilate(closed_img, kernel, iterations=1)

    # 골격화
    enhanced_img = skeletonize(dilated_img // 255).astype(np.uint8) * 255

    # Minutiae 추출
    minutiae_end = []
    minutiae_bif = []

    # 8방향 탐색 및 필터링
    # 8방향 탐색 및 필터링
    for i in range(1, enhanced_img.shape[0] - 1):
        for j in range(1, enhanced_img.shape[1] - 1):
            if enhanced_img[i, j] == 0:
                # 3x3 윈도우 설정
                window = enhanced_img[i-1:i+2, j-1:j+2]
                count = np.sum(window == 0)

                # Minutiae end point 검사 (주변 픽셀 개수 필터링)
                if count == 2:  # 3x3 윈도우 중 2개의 값만 0인 경우
                    minutiae_end.append((j, i))

                # Minutiae bifurcation point 검사 (주변 픽셀 개수 필터링)
                elif count == 4:  # 3x3 윈도우 중 4개의 값이 0인 경우
                    # 추가 조건: 세 방향으로 가지가 나뉘는 지점 확인
                    neighbors = [
                        window[0, 1], window[2, 1],  # 위, 아래
                        window[1, 0], window[1, 2],  # 왼쪽, 오른쪽
                        window[0, 0], window[0, 2],  # 좌상단, 우상단
                        window[2, 0], window[2, 2]   # 좌하단, 우하단
                    ]
                    if neighbors.count(0) == 3:
                        # 교차되는 선의 길이가 2 이상인 경우에만 bifurcation으로 간주
                        vertical_cross = np.sum(enhanced_img[i-1:i+2, j] == 0)
                        horizontal_cross = np.sum(enhanced_img[i, j-1:j+2] == 0)
                        diagonal_cross_1 = np.sum([enhanced_img[i-1, j-1], enhanced_img[i+1, j+1]] == 0)
                        diagonal_cross_2 = np.sum([enhanced_img[i-1, j+1], enhanced_img[i+1, j-1]] == 0)
                        if (vertical_cross >= 2 and horizontal_cross >= 2) or \
                           (diagonal_cross_1 >= 2 and horizontal_cross >= 2) or \
                           (diagonal_cross_2 >= 2 and horizontal_cross >= 2) or \
                           (vertical_cross >= 2 and diagonal_cross_1 >= 2) or \
                           (vertical_cross >= 2 and diagonal_cross_2 >= 2):
                            minutiae_bif.append((j, i))


    # 이진화된 이미지 위에 Minutiae 표시된 이미지 생성
    img_with_minutiae = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)

    # Minutiae end point 표시 (파란색)
    for end in minutiae_end:
        cv2.circle(img_with_minutiae, end, 3, (0, 0, 255), -1)

    # Minutiae bifurcation point 표시 (빨간색)
    for bif in minutiae_bif:
        cv2.circle(img_with_minutiae, bif, 3, (255, 0, 0), -1)

    if flg_show==True:
        # 이미지 출력
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(enhanced_img, cmap='gray')
        plt.title('Skeleton')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(img_with_minutiae)
        plt.title('Minutiae')
        plt.axis('off')

        plt.show()

    return minutiae_end, minutiae_bif


def calculate_distance(points1, points2):
    # 가장 가까운 특징점 쌍의 거리 합 계산
    dist_sum = 0
    for pt1 in points1:
        min_dist = np.inf
        for pt2 in points2:
            dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            if dist < min_dist:
                min_dist = dist
        dist_sum += min_dist
    return dist_sum

def match_fingerprints(test_img_path, list_train):
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    test_end, test_bif = get_fp_feature(test_img, flg_show=False)
    
    best_match = None
    min_distance = np.inf
    best_train_end, best_train_bif = None, None
    
    for train_img_path in list_train:
        train_img = cv2.imread(train_img_path, cv2.IMREAD_GRAYSCALE)
        train_end, train_bif = get_fp_feature(train_img, flg_show=False)
        
        # 거리 계산 (end 포인트와 bifurcation 포인트 모두 사용)
        distance = calculate_distance(test_end, train_end) + calculate_distance(test_bif, train_bif)
        
        if distance < min_distance:
            min_distance = distance
            best_match = train_img_path
            best_train_end, best_train_bif = train_end, train_bif
    
    return best_match, test_end, test_bif, best_train_end, best_train_bif, min_distance


def draw_matched_minutiae(test_img, train_img, test_end, test_bif, train_end, train_bif):
    # 이미지 출력 및 연결 선 그리기
    plt.figure(figsize=(10, 10))

    # 테스트 이미지
    plt.subplot(2, 2, 1)
    plt.imshow(test_img, cmap='gray')
    plt.title('Test Image')
    plt.axis('off')

    # 훈련 이미지
    plt.subplot(2, 2, 2)
    plt.imshow(train_img, cmap='gray')
    plt.title('Train Image')
    plt.axis('off')

    # 테스트 이미지 minutiae 표시
    plt.subplot(2, 2, 3)
    img_with_test_minutiae = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    for end in test_end:
        cv2.circle(img_with_test_minutiae, end, 3, (0, 0, 255), -1)
    for bif in test_bif:
        cv2.circle(img_with_test_minutiae, bif, 3, (255, 0, 0), -1)
    plt.imshow(img_with_test_minutiae)
    plt.title('Test Minutiae')
    plt.axis('off')

    # 훈련 이미지 minutiae 표시 및 연결 선
    plt.subplot(2, 2, 4)
    img_with_train_minutiae = cv2.cvtColor(train_img, cv2.COLOR_GRAY2BGR)
    for end in train_end:
        cv2.circle(img_with_train_minutiae, end, 3, (0, 0, 255), -1)
    for bif in train_bif:
        cv2.circle(img_with_train_minutiae, bif, 3, (255, 0, 0), -1)
    for i, end in enumerate(test_end):
        for j, train_end in enumerate(train_end):
            if end == train_end:
                cv2.line(img_with_train_minutiae, (end[0], end[1]), (train_end[0], train_end[1]), (0, 255, 0), thickness=2)
    for i, bif in enumerate(test_bif):
        for j, train_bif in enumerate(train_bif):
            if bif == train_bif:
                cv2.line(img_with_train_minutiae, (bif[0], bif[1]), (train_bif[0], train_bif[1]), (0, 255, 0), thickness=2)
    plt.imshow(img_with_train_minutiae)
    plt.title('Train Minutiae with Connections')
    plt.axis('off')

    plt.tight_layout()
    plt.show()