import random

import cv2
import numpy as np

from .utilities import CANVAS_SIZE, labels_bboxes_similarity, labels_similarity
from typing import Any


class ExemplarSelection:
    """
    훈련 데이터에서 테스트 데이터와 유사한 예시(exemplar)들을 선택하는 기본 클래스
    
    Few-shot learning에서 사용할 예시 데이터를 선택하는 역할을 수행합니다.
    프롬프트에 포함할 가장 적절한 훈련 예시들을 유사도 기반으로 선택합니다.
    """
    
    def __init__(
        self,
        train_data: list,       # 훈련 데이터 리스트 (예시 선택의 후보군)
        candidate_size: int,    # 후보 데이터 크기 (-1이면 전체 사용, 양수면 해당 개수만큼 샘플링)
        num_prompt: int,        # 최종적으로 선택할 예시의 개수 (프롬프트에 포함될 예시 수)
        shuffle: bool = True,   # 선택된 예시들을 셞플할지 여부
    ):
        """
        ExemplarSelection 초기화
        
        Args:
            train_data: 전체 훈련 데이터 리스트
            candidate_size: 후보군 크기 (메모리 효율성을 위해 전체 데이터의 일부만 사용)
            num_prompt: 최종 선택할 예시 개수 (보통 5-10개)
            shuffle: 선택된 예시들의 순서를 무작위로 섞을지 여부
        """
        self.train_data = train_data            # 훈련 데이터 저장
        self.candidate_size = candidate_size    # 후보군 크기
        self.num_prompt = num_prompt            # 선택할 예시 개수
        self.shuffle = shuffle                  # 셔플 여부
        
        # 후보 데이터 크기 제한 (메모리 및 계산 효율성)
        if self.candidate_size > 0:
            random.shuffle(self.train_data)                              # 전체 데이터를 무작위로 섞음
            self.train_data = self.train_data[: self.candidate_size]     # 지정된 크기만큼만 사용

    def __call__(self, test_data: dict):
        """
        테스트 데이터에 대해 가장 유사한 예시들을 선택
        
        Args:
            test_data: 예시를 찾을 대상이 되는 테스트 데이터
            
        Returns:
            선택된 예시들의 리스트
            
        Note:
            이 메서드는 하위 클래스에서 구체적으로 구현되어야 함
        """
        pass

    def _is_filter(self, data):
        """
        데이터가 필터링되어야 하는지 확인 (유효하지 않은 데이터 제거)
        
        Args:
            data: 검사할 데이터 딕셔너리
            
        Returns:
            bool: True면 필터링 대상 (제외), False면 유효한 데이터
            
        Logic:
            바운딩박스의 width, height가 0인 요소가 있는지 확인
            - data["discrete_gold_bboxes"][:, 2:]: 모든 요소의 [width, height] 부분
            - == 0: width 또는 height가 0인지 확인
            - .sum().bool().item(): 하나라도 0이 있으면 True 반환
        """
        return (data["discrete_gold_bboxes"][:, 2:] == 0).sum().bool().item()

    def _retrieve_exemplars(self, scores: list):
        """
        유사도 점수를 바탕으로 최종 예시들을 선택
        
        Args:
            scores: [(인덱스, 유사도점수), ...] 형태의 리스트
            
        Returns:
            선택된 예시 데이터들의 리스트
            
        Process:
            1. 유사도 점수 기준 내림차순 정렬 (가장 유사한 것부터)
            2. 유효한 데이터만 선택 (필터링된 데이터 제외)
            3. 지정된 개수만큼 선택
            4. 필요시 순서 셔플
        """
        # 유사도 점수 기준으로 내림차순 정렬 (점수가 높을수록 유사함)
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        exemplars = []  # 선택된 예시들을 저장할 리스트
        
        # 점수가 높은 순서대로 순회하면서 유효한 데이터 선택
        for i in range(len(self.train_data)):
            data_idx = scores[i][0]  # 해당 순위의 데이터 인덱스
            
            # 유효하지 않은 데이터는 건너뛰기 (바운딩박스가 0인 경우 등)
            if not self._is_filter(self.train_data[data_idx]):
                exemplars.append(self.train_data[data_idx])  # 유효한 데이터 추가
                
                # 필요한 개수만큼 선택되면 종료
                if len(exemplars) == self.num_prompt:
                    break
        
        # 선택된 예시들의 순서를 무작위로 섞기 (편향 방지)
        if self.shuffle:
            random.shuffle(exemplars)
            
        return exemplars


"""--------------------- Un-Used Selectors ---------------------"""

class GenTypeExemplarSelection(ExemplarSelection):
    def __call__(self, test_data: dict):
        scores = []
        test_labels = test_data["labels"]
        for i in range(len(self.train_data)):
            train_labels = self.train_data[i]["labels"]
            score = labels_similarity(train_labels, test_labels)
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


class GenTypeSizeExemplarSelection(ExemplarSelection):
    labels_weight = 0.5
    bboxes_weight = 0.5

    def __call__(self, test_data: dict):
        scores = []
        test_labels = test_data["labels"]
        test_bboxes = test_data["bboxes"][:, 2:]
        for i in range(len(self.train_data)):
            train_labels = self.train_data[i]["labels"]
            train_bboxes = self.train_data[i]["bboxes"][:, 2:]
            score = labels_bboxes_similarity(
                train_labels,
                train_bboxes,
                test_labels,
                test_bboxes,
                self.labels_weight,
                self.bboxes_weight,
            )
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


class GenRelationExemplarSelection(ExemplarSelection):
    def __call__(self, test_data: dict):
        scores = []
        test_labels = test_data["labels"]
        for i in range(len(self.train_data)):
            train_labels = self.train_data[i]["labels"]
            score = labels_similarity(train_labels, test_labels)
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


class CompletionExemplarSelection(ExemplarSelection):
    labels_weight = 0.0
    bboxes_weight = 1.0

    def __call__(self, test_data: dict):
        scores = []
        test_labels = test_data["labels"][:1]
        test_bboxes = test_data["bboxes"][:1, :]
        for i in range(len(self.train_data)):
            train_labels = self.train_data[i]["labels"][:1]
            train_bboxes = self.train_data[i]["bboxes"][:1, :]
            score = labels_bboxes_similarity(
                train_labels,
                train_bboxes,
                test_labels,
                test_bboxes,
                self.labels_weight,
                self.bboxes_weight,
            )
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


class ContentAwareExemplarSelection(ExemplarSelection):
    canvas_width, canvas_height = CANVAS_SIZE["posterlayout"]

    def _to_binary_image(self, content_bboxes):
        binary_image = np.zeros((self.canvas_height, self.canvas_width), dtype=np.uint8)
        content_bboxes = content_bboxes.tolist()
        for content_bbox in content_bboxes:
            l, t, w, h = content_bbox
            cv2.rectangle(binary_image, (l, t), (l + w, t + h), (255,), thickness=-1)
        return binary_image

    def __call__(self, test_data: dict):
        scores = []
        test_content_bboxes = test_data["discrete_content_bboxes"]
        test_binary = self._to_binary_image(test_content_bboxes)
        for i in range(len(self.train_data)):
            train_content_bboxes = self.train_data[i]["discrete_content_bboxes"]
            train_binary = self._to_binary_image(train_content_bboxes)
            intersection = cv2.bitwise_and(train_binary, test_binary)
            union = cv2.bitwise_or(train_binary, test_binary)
            iou = (np.sum(intersection) + 1) / (np.sum(union) + 1)
            scores.append([i, iou])
        return self._retrieve_exemplars(scores)





"""--------------------- Used Selectors ---------------------"""

class RefinementExemplarSelection(ExemplarSelection):
    """
    레이아웃 리파인먼트 작업을 위한 예시 선택 클래스
    
    레이블(UI 요소 타입)과 바운딩박스(위치/크기) 정보를 모두 고려하여
    테스트 데이터와 가장 유사한 훈련 예시들을 선택합니다.
    기존 레이아웃을 개선하거나 수정하는 작업에 적합합니다.
    """
    
    # 유사도 계산 시 각 요소의 가중치 (합이 1.0이 되도록 설정)
    labels_weight = 0.5   # 레이블 유사도 가중치 (UI 요소 타입의 중요도)
    bboxes_weight = 0.5   # 바운딩박스 유사도 가중치 (위치/크기의 중요도)

    def __call__(self, test_data: dict):
        """
        테스트 데이터와 유사한 훈련 예시들을 선택
        
        Args:
            test_data: 테스트 데이터 딕셔너리
                      - "labels": UI 요소 타입들의 텐서
                      - "bboxes": 바운딩박스 좌표들의 텐서
        
        Returns:
            선택된 예시들의 리스트 (유사도 기준 상위 num_prompt개)
            
        Process:
            1. 각 훈련 데이터와 테스트 데이터 간의 복합 유사도 계산
            2. 레이블 유사도와 바운딩박스 유사도를 가중 평균으로 결합
            3. 유사도 점수 기준으로 최적의 예시들 선택
        """
        scores = []  # [(인덱스, 유사도점수)] 리스트
        
        # 테스트 데이터에서 비교 대상 추출
        test_labels = test_data["labels"]    # 테스트 레이아웃의 UI 요소 타입들
        test_bboxes = test_data["bboxes"]    # 테스트 레이아웃의 바운딩박스들
        
        # 모든 훈련 데이터와 유사도 계산
        for i in range(len(self.train_data)):
            # 각 훈련 예시에서 비교 대상 추출
            train_labels = self.train_data[i]["labels"]    # 훈련 레이아웃의 UI 요소 타입들
            train_bboxes = self.train_data[i]["bboxes"]    # 훈련 레이아웃의 바운딩박스들
            
            # 레이블과 바운딩박스를 모두 고려한 복합 유사도 계산
            # labels_bboxes_similarity: 두 레이아웃 간의 전체적인 유사도를 계산
            # - 레이블 유사도: UI 요소 타입의 일치도
            # - 바운딩박스 유사도: 위치와 크기의 일치도
            # - 가중치를 적용하여 두 유사도를 결합
            score = labels_bboxes_similarity(
                train_labels,           # 훈련 데이터의 레이블
                train_bboxes,          # 훈련 데이터의 바운딩박스
                test_labels,           # 테스트 데이터의 레이블
                test_bboxes,           # 테스트 데이터의 바운딩박스
                self.labels_weight,    # 레이블 가중치 (0.5)
                self.bboxes_weight,    # 바운딩박스 가중치 (0.5)
            )
            scores.append([i, score])  # (인덱스, 유사도) 쌍 저장
        
        # 유사도 점수를 바탕으로 최종 예시들 선택 및 반환
        return self._retrieve_exemplars(scores)


class TextToLayoutExemplarSelection(ExemplarSelection):
    """
    텍스트-투-레이아웃 생성을 위한 예시 선택 클래스
    
    CLIP 텍스트 임베딩을 사용하여 의미적으로 유사한 텍스트를 가진
    훈련 예시들을 선택합니다. 사용자의 텍스트 쿼리와 가장 유사한
    의미를 가진 레이아웃 예시들을 찾는 데 사용됩니다.
    """
    
    def __call__(self, test_data: dict):
        """
        텍스트 임베딩 유사도를 기반으로 예시들을 선택
        
        Args:
            test_data: 테스트 데이터 딕셔너리
                      - "embedding": CLIP으로 인코딩된 텍스트 임베딩 벡터
        
        Returns:
            의미적으로 유사한 예시들의 리스트 (코사인 유사도 기준 상위 num_prompt개)
            
        Process:
            1. 테스트 텍스트의 CLIP 임베딩과 각 훈련 데이터의 임베딩 간 코사인 유사도 계산
            2. 의미적 유사도가 높은 순서대로 정렬
            3. 가장 유사한 의미를 가진 예시들 선택
            
        Note:
            CLIP 임베딩은 이미 정규화되어 있어 내적 연산이 코사인 유사도와 동일함
        """
        scores = []  # [(인덱스, 유사도점수)] 리스트
        
        # 테스트 데이터의 텍스트 임베딩 추출
        test_embedding = test_data["embedding"]  # CLIP으로 인코딩된 테스트 텍스트 벡터
        
        # 모든 훈련 데이터와 의미적 유사도 계산
        for i in range(len(self.train_data)):
            # 각 훈련 예시의 텍스트 임베딩 추출
            train_embedding = self.train_data[i]["embedding"]  # 훈련 데이터의 CLIP 임베딩
            
            # Ensure both embeddings are on the same device
            device = test_embedding.device  # Get the device of the test embedding
            train_embedding = train_embedding.to(device)  # Move train embedding to the same device

            # 코사인 유사도 계산 (내적 연산)
            # @ 연산자: 행렬 곱셈 (내적)
            # .T: 전치 행렬 (벡터를 올바른 차원으로 맞춤)
            # .item(): 텐서에서 스칼라 값 추출
            # CLIP 임베딩은 정규화되어 있어 내적 = 코사인 유사도
            score = (train_embedding @ test_embedding.T).item()
            scores.append([i, score])  # (인덱스, 유사도) 쌍 저장
        
        # 의미적 유사도 점수를 바탕으로 최종 예시들 선택 및 반환
        return self._retrieve_exemplars(scores)


SELECTOR_MAP = {
    # Un-Used Selectors
    "gent": GenTypeExemplarSelection,
    "gents": GenTypeSizeExemplarSelection,
    "genr": GenRelationExemplarSelection,
    "completion": CompletionExemplarSelection,
    "content": ContentAwareExemplarSelection,

    # Used Selectors
    "refinement": RefinementExemplarSelection,
    "text": TextToLayoutExemplarSelection,
}


def create_selector(task: str, train_data: list, candidate_size: int = -1, num_prompt: int = 5, *args, **kwargs) -> ExemplarSelection:
    """
    주어진 task에 맞는 ExemplarSelection 객체를 생성합니다.

    Args:
        task (str): 사용할 selector의 종류. 기본값은 "text".
        train_data (list): 예시 선택에 사용할 훈련 데이터 리스트. (Proceed data: 각 데이터는 딕셔너리 형태로 레이블, 바운딩 박스 등을 포함).
        candidate_size (int): 후보 데이터의 크기. -1이면 전체 사용, 양수면 해당 개수만큼 샘플링. 기본값 -1.
        num_prompt (int): 최종적으로 선택할 예시의 개수 (프롬프트에 포함될 예시의 개수). 기본값 5.
        *args: selector 클래스에 전달할 추가 positional arguments.
        **kwargs: selector 클래스에 전달할 추가 keyword arguments.

    Returns:
        ExemplarSelection: 선택된 selector 객체.
    """
    selector_cls = SELECTOR_MAP[task]
    selector = selector_cls(
        train_data=train_data, 
        candidate_size=candidate_size, 
        num_prompt=num_prompt,
        *args,
        **kwargs,
    )
    return selector

"""
🎯 두 가지 Selector의 선택 기준:
    RefinementExemplarSelection:
        - 레이블 + 바운딩박스 구조적 유사도
        - 기존 데이터 그대로 반환

    TextToLayoutExemplarSelection:
        - CLIP 임베딩 의미적 유사도
        - 기존 데이터 그대로 반환
"""