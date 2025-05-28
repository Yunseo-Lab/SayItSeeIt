"""
Parser 모듈 - 언어 모델의 출력을 구조화된 레이아웃 데이터로 변환

이 모듈은 생성형 언어 모델이 생성한 텍스트 형태의 레이아웃 정보를 
실제 렌더링 가능한 구조화된 데이터(레이블과 바운딩박스)로 변환합니다.

주요 기능:
1. HTML 형식의 레이아웃 텍스트 파싱
2. 시퀀스 형식의 레이아웃 텍스트 파싱  
3. 정규화된 좌표를 픽셀 좌표로 역변환
4. 에러 처리를 통한 robust한 파싱

입력/출력 예시:
- 입력: "title 0 50 10 300 40 | image 1 20 60 200 150"
- 출력: (labels=[0,1], bboxes=[[0.05,0.01,0.3,0.04], [0.02,0.06,0.2,0.15]])
"""

import re         # 정규표현식 모듈 - HTML/시퀀스 텍스트에서 패턴 추출용
import torch      # PyTorch 텐서 연산 라이브러리
from typing import List, Tuple, Optional

from .utilities import CANVAS_SIZE, ID2LABEL


class Parser:
    """
    언어 모델이 생성한 레이아웃 텍스트를 파싱하여 구조화된 데이터로 변환하는 클래스
    
    이 클래스는 Serializer의 역방향 작업을 수행합니다:
    - Serializer: 구조화된 데이터 → 텍스트 (모델 입력용)
    - Parser: 텍스트 → 구조화된 데이터 (모델 출력 해석용)
    
    지원하는 출력 형식:
    1. seq: 시퀀스 형식 - "label id x y w h | label id x y w h"
       예시: "title 0 50 10 300 40 | image 1 20 60 200 150"
    
    2. html: HTML 형식 - '<div class="label" style="position: absolute; left: Xpx; top: Ypx; width: Wpx; height: Hpx;"></div>'
       예시: '<div class="title" style="left: 50px; top: 10px; width: 300px; height: 40px;"></div>'
    
    좌표 정규화:
    - 입력: 픽셀 좌표 (0 ~ canvas_width/height)
    - 출력: 정규화된 좌표 (0.0 ~ 1.0)
    """
    
    def __init__(self, dataset: str, output_format: str):
        """
        Parser 초기화
        
        Args:
            dataset (str): 데이터셋 이름 ('rico', 'publaynet', 'wireframe' 등)
            output_format (str): 출력 형식 ('seq' 또는 'html')
        """
        self.dataset = dataset
        self.output_format = output_format
        
        # 레이블 매핑 정보 설정
        self.id2label = ID2LABEL[self.dataset]  # {0: 'title', 1: 'image', ...}
        self.label2id = {v: k for k, v in self.id2label.items()}  # {'title': 0, 'image': 1, ...}
        
        # 캔버스 크기 설정 (정규화용)
        self.canvas_width, self.canvas_height = CANVAS_SIZE[self.dataset]

    def _extract_labels_and_bboxes(self, prediction: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        예측 텍스트에서 레이블과 바운딩박스를 추출하는 통합 메서드
        
        출력 형식에 따라 적절한 파싱 메서드를 호출합니다.
        
        Args:
            prediction (str): 언어 모델이 생성한 레이아웃 텍스트
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - labels: 레이블 ID 텐서 (shape: [N])
                - bboxes: 정규화된 바운딩박스 텐서 (shape: [N, 4]) [x, y, w, h]
        
        Raises:
            RuntimeError: 파싱 과정에서 오류 발생 시
        """
        if self.output_format == "seq":
            return self._extract_labels_and_bboxes_from_seq(prediction)
        elif self.output_format == "html":
            return self._extract_labels_and_bboxes_from_html(prediction)
        else:
            raise ValueError(f"지원하지 않는 출력 형식: {self.output_format}")

    def _extract_labels_and_bboxes_from_html(self, prediction: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HTML 형식의 예측 텍스트에서 레이블과 바운딩박스를 추출
        
        HTML 형식 예시:
        '<div class="canvas" style="..."></div>
         <div class="title" style="left: 50px; top: 10px; width: 300px; height: 40px;"></div>
         <div class="image" style="left: 20px; top: 60px; width: 200px; height: 150px;"></div>'
        
        파싱 과정:
        1. 정규표현식으로 클래스명(레이블) 추출
        2. CSS 스타일에서 left, top, width, height 값 추출
        3. 픽셀 좌표를 정규화된 좌표로 변환
        4. 캔버스 div는 제외하고 실제 UI 요소만 처리
        
        Args:
            prediction (str): HTML 형식의 레이아웃 텍스트
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - labels: 레이블 ID 텐서
                - bboxes: 정규화된 바운딩박스 텐서 [x, y, w, h]
        
        Raises:
            RuntimeError: 추출된 레이블과 좌표 정보의 개수가 일치하지 않을 때
        """
        # 1. 클래스명(레이블) 추출 - 첫 번째는 canvas이므로 제외
        labels = re.findall('<div class="(.*?)"', prediction)[1:]  # canvas 제거
        
        # 2. CSS 좌표 정보 추출 - 첫 번째는 canvas이므로 제외
        x = re.findall(r"left:.?(\d+)px", prediction)[1:]
        y = re.findall(r"top:.?(\d+)px", prediction)[1:]
        w = re.findall(r"width:.?(\d+)px", prediction)[1:]
        h = re.findall(r"height:.?(\d+)px", prediction)[1:]
        
        # 3. 데이터 일관성 검증
        if not (len(labels) == len(x) == len(y) == len(w) == len(h)):
            raise RuntimeError(
                f"HTML 파싱 오류: 데이터 길이 불일치 - "
                f"레이블: {len(labels)}, x: {len(x)}, y: {len(y)}, w: {len(w)}, h: {len(h)}"
            )
        
        # 4. 레이블을 ID로 변환
        labels = torch.tensor([self.label2id[label] for label in labels])
        
        # 5. 픽셀 좌표를 정규화된 좌표로 변환
        # 최소 길이를 사용하여 안전하게 처리
        min_len = min(len(labels), len(x), len(y), len(w), len(h))
        bboxes = torch.tensor(
            [
                [
                    int(x[i]) / self.canvas_width,   # x 정규화
                    int(y[i]) / self.canvas_height,  # y 정규화
                    int(w[i]) / self.canvas_width,   # width 정규화
                    int(h[i]) / self.canvas_height,  # height 정규화
                ]
                for i in range(min_len)
            ]
        )
        return labels, bboxes

    def _extract_labels_and_bboxes_from_seq(self, prediction: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        시퀀스 형식의 예측 텍스트에서 레이블과 바운딩박스를 추출
        
        시퀀스 형식 예시:
        "title 0 50 10 300 40 | image 1 20 60 200 150"
        
        형식 구조:
        "label_name label_id x y width height | label_name label_id x y width height"
        
        파싱 과정:
        1. 정규표현식으로 패턴 매칭
        2. 레이블명을 ID로 변환
        3. 픽셀 좌표를 정규화된 좌표로 변환
        
        Args:
            prediction (str): 시퀀스 형식의 레이아웃 텍스트
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - labels: 레이블 ID 텐서
                - bboxes: 정규화된 바운딩박스 텐서 [x, y, w, h]
        """
        # 1. 가능한 레이블 집합 구성
        label_set = list(self.label2id.keys())
        
        # 2. 시퀀스 패턴 정의: "label_name id x y w h"
        seq_pattern = r"(" + "|".join(label_set) + r") (\d+) (\d+) (\d+) (\d+)"
        
        # 3. 정규표현식으로 모든 패턴 추출
        res = re.findall(seq_pattern, prediction)
        
        # 4. 레이블명을 ID로 변환
        labels = torch.tensor([self.label2id[item[0]] for item in res])
        
        # 5. 픽셀 좌표를 정규화된 좌표로 변환
        # item[1]은 레이블 ID이므로 건너뛰고, item[2:6]이 x,y,w,h
        bboxes = torch.tensor(
            [
                [
                    int(item[1]) / self.canvas_width,   # x 정규화
                    int(item[2]) / self.canvas_height,  # y 정규화
                    int(item[3]) / self.canvas_width,   # width 정규화
                    int(item[4]) / self.canvas_height,  # height 정규화
                ]
                for item in res
            ]
        )
        return labels, bboxes

    def __call__(self, predictions: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        배치 예측 텍스트를 파싱하여 구조화된 데이터로 변환
        
        이 메서드는 Parser의 main entry point로, 여러 개의 예측 텍스트를
        한 번에 처리합니다. 각 예측에 대해 개별적으로 파싱을 시도하며,
        파싱에 실패한 경우 해당 예측을 건너뛰고 계속 진행합니다.
        
        에러 처리 전략:
        - 개별 예측 파싱 실패 시 해당 예측만 건너뛰고 계속 진행
        - 전체 배치 처리가 중단되지 않도록 robust하게 설계
        - 파싱 가능한 예측들만 결과에 포함
        
        사용 예시:
        ```python
        parser = Parser('rico', 'seq')
        predictions = [
            "title 0 50 10 300 40 | image 1 20 60 200 150",
            "invalid prediction text",  # 이것은 건너뛰어짐
            "button 2 100 200 80 30"
        ]
        results = parser(predictions)  # 2개의 성공적인 파싱 결과 반환
        ```
        
        Args:
            predictions (List[str]): 언어 모델이 생성한 레이아웃 텍스트 리스트
        
        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: 성공적으로 파싱된 결과 리스트
                각 튜플은 (labels, bboxes) 형태:
                - labels: 레이블 ID 텐서 (shape: [N])
                - bboxes: 정규화된 바운딩박스 텐서 (shape: [N, 4])
        
        Note:
            - 파싱에 실패한 예측은 자동으로 제외되므로, 
              입력과 출력의 길이가 다를 수 있습니다.
            - 모든 예측이 파싱에 실패한 경우 RuntimeError가 발생합니다.
        
        Raises:
            RuntimeError: 모든 예측이 파싱에 실패한 경우
        """
        parsed_predictions = []
        parsing_errors = []  # 파싱 에러 추적용
        
        for i, prediction in enumerate(predictions):
            try:
                # 개별 예측 파싱 시도
                labels, bboxes = self._extract_labels_and_bboxes(prediction)
                parsed_predictions.append((labels, bboxes))
                
            except Exception as e:
                # 파싱 실패 시 해당 예측 건너뛰기
                parsing_errors.append(f"예측 {i} 파싱 실패: {e}")
                # 디버깅이 필요한 경우 아래 주석을 해제
                # print(f"예측 {i} 파싱 실패: {e}")
                continue
        
        # 모든 예측이 파싱 실패한 경우 에러 발생
        if not parsed_predictions:
            error_msg = f"모든 예측({len(predictions)}개)의 파싱에 실패했습니다."
            if parsing_errors:
                error_msg += f"\n파싱 에러 목록:\n" + "\n".join(parsing_errors[:5])  # 최대 5개까지만 표시
                if len(parsing_errors) > 5:
                    error_msg += f"\n... 및 {len(parsing_errors) - 5}개의 추가 에러"
            raise RuntimeError(error_msg)
                
        return parsed_predictions
