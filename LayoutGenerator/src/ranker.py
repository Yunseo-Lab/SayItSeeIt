# ranker.py - 레이아웃 예측 결과의 품질을 평가하고 순위를 매기는 모듈
import torch

from .utilities import (
    compute_alignment,      # 정렬 점수 계산 함수
    compute_maximum_iou,    # 최대 IoU 점수 계산 함수 
    compute_overlap,        # 겹침 점수 계산 함수
    convert_ltwh_to_ltrb,   # 좌표 형식 변환 함수 (left-top-width-height -> left-top-right-bottom)
    read_pt,                # PyTorch 파일 읽기 함수
)


class Ranker:
    """
    레이아웃 예측 결과들을 품질에 따라 순위를 매기는 클래스
    
    다음 지표들을 종합적으로 고려하여 레이아웃의 품질을 평가합니다:
    1. 정렬 점수 (alignment): 요소들이 얼마나 잘 정렬되어 있는지
    2. 겹침 점수 (overlap): 요소들 간의 겹침 정도
    3. IoU 점수 (optional): 검증 데이터와의 유사도
    """
    
    # 품질 계산을 위한 가중치 파라미터들
    lambda_1 = 0.2  # 정렬 점수 가중치
    lambda_2 = 0.2  # 겹침 점수 가중치  
    lambda_3 = 0.6  # IoU 점수 가중치 (검증 데이터가 있을 때만 사용)

    def __init__(self, val_path=None):
        """
        Ranker 클래스 초기화
        
        Args:
            val_path (str, optional): 검증 데이터 파일 경로
                                     None이면 IoU 점수는 계산하지 않음
        """
        self.val_path = val_path
        
        # 검증 데이터가 제공된 경우 로드하여 저장
        if self.val_path:
            self.val_data = read_pt(val_path)  # PyTorch 파일에서 검증 데이터 읽기
            # 검증 데이터에서 라벨과 바운딩 박스 정보 추출
            self.val_labels = [vd["labels"] for vd in self.val_data]
            self.val_bboxes = [vd["bboxes"] for vd in self.val_data]

    def __call__(self, predictions: list):
        """
        예측 결과들을 품질에 따라 순위를 매기는 메인 함수
        
        Args:
            predictions (list): 예측 결과 리스트
                              각 원소는 (pred_labels, pred_bboxes) 튜플 형태
        
        Returns:
            list: 품질 점수가 낮은 순으로 정렬된 예측 결과 리스트
                 (낮은 점수 = 더 좋은 품질)
        """
        metrics = []  # 각 예측에 대한 지표들을 저장할 리스트
        
        # 각 예측 결과에 대해 품질 지표들을 계산
        for pred_labels, pred_bboxes in predictions:
            metric = []  # 현재 예측의 지표들을 저장할 리스트
            
            # 텐서 차원 맞추기 (배치 차원 추가)
            _pred_labels = pred_labels.unsqueeze(0)
            _pred_bboxes = convert_ltwh_to_ltrb(pred_bboxes).unsqueeze(0)  # 좌표 형식 변환 후 배치 차원 추가
            _pred_padding_mask = torch.ones_like(_pred_labels).bool()  # 패딩 마스크 생성 (모든 요소가 유효함을 표시)
            
            # 1. 정렬 점수 계산 - 요소들이 얼마나 잘 정렬되어 있는지 평가
            metric.append(compute_alignment(_pred_bboxes, _pred_padding_mask))
            
            # 2. 겹침 점수 계산 - 요소들 간의 겹침 정도 평가
            metric.append(compute_overlap(_pred_bboxes, _pred_padding_mask))
            
            # 3. IoU 점수 계산 (검증 데이터가 있는 경우만)
            if self.val_path:
                metric.append(
                    compute_maximum_iou(
                        pred_labels,      # 예측 라벨
                        pred_bboxes,      # 예측 바운딩 박스
                        self.val_labels,  # 검증 라벨
                        self.val_bboxes,  # 검증 바운딩 박스
                    )
                )
            
            metrics.append(metric)  # 현재 예측의 모든 지표를 전체 리스트에 추가

        # 지표들을 텐서로 변환하여 계산 최적화
        metrics = torch.tensor(metrics)
        
        # 정규화를 위해 각 지표별 최솟값과 최댓값 계산
        min_vals, _ = torch.min(metrics, 0, keepdim=True)  # 각 지표의 최솟값
        max_vals, _ = torch.max(metrics, 0, keepdim=True)  # 각 지표의 최댓값
        
        # Min-Max 정규화: 모든 지표를 0~1 범위로 스케일링
        scaled_metrics = (metrics - min_vals) / (max_vals - min_vals)
        
        # 가중치를 적용하여 최종 품질 점수 계산
        if self.val_path:
            # 검증 데이터가 있는 경우: 3가지 지표 모두 사용
            quality = (
                scaled_metrics[:, 0] * self.lambda_1      # 정렬 점수 * 가중치
                + scaled_metrics[:, 1] * self.lambda_2    # 겹침 점수 * 가중치  
                + (1 - scaled_metrics[:, 2]) * self.lambda_3  # (1 - IoU점수) * 가중치 (IoU는 높을수록 좋으므로 1에서 빼줌)
            )
        else:
            # 검증 데이터가 없는 경우: 정렬과 겹침 점수만 사용
            quality = (
                scaled_metrics[:, 0] * self.lambda_1      # 정렬 점수 * 가중치
                + scaled_metrics[:, 1] * self.lambda_2    # 겹침 점수 * 가중치
            )
        
        # 품질 점수에 따라 예측 결과들을 오름차순 정렬 (낮은 점수 = 더 좋은 품질)
        _predictions = sorted(zip(predictions, quality), key=lambda x: x[1].item())
        
        # 정렬된 예측 결과만 추출하여 반환
        ranked_predictions = [item[0] for item in _predictions]
        return ranked_predictions
