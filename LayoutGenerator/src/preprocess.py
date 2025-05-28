import copy
import random

import cv2
import torch
import torchvision.transforms as T
from pandas import DataFrame

from .transforms import (
    AddCanvasElement,
    AddGaussianNoise,
    AddRelation,
    CLIPTextEncoder,
    DiscretizeBoundingBox,
    LabelDictSort,
    LexicographicSort,
    SaliencyMapToBBoxes,
    ShuffleElements,
)
from .utilities import CANVAS_SIZE, ID2LABEL, clean_text


class Processor:
    # 기본 반환 키들 (서브클래스에서 오버라이드 가능)
    return_keys = []
    
    def __init__(
        self, index2label: dict, canvas_width: int, canvas_height: int, *args, **kwargs
    ):
        self.index2label = index2label
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.sort_by_pos = kwargs.get("sort_by_pos", None)
        self.shuffle_before_sort_by_label = kwargs.get(
            "shuffle_before_sort_by_label", None
        )
        self.sort_by_pos_before_sort_by_label = kwargs.get(
            "sort_by_pos_before_sort_by_label", None
        )

        if not any(
            [
                self.sort_by_pos,
                self.shuffle_before_sort_by_label,
                self.sort_by_pos_before_sort_by_label,
            ]
        ):
            raise ValueError(
                "At least one of sort_by_pos, shuffle_before_sort_by_label, or sort_by_pos_before_sort_by_label must be True."
            )
        self.transform_functions = self._config_base_transform()
        # transform 파이프라인 초기화
        self.transform = T.Compose(self.transform_functions)

    def _config_base_transform(self):
        transform_functions = list()
        if self.sort_by_pos:
            transform_functions.append(LexicographicSort())
        else:
            if self.shuffle_before_sort_by_label:
                transform_functions.append(ShuffleElements())
            elif self.sort_by_pos_before_sort_by_label:
                transform_functions.append(LexicographicSort())
            transform_functions.append(LabelDictSort(self.index2label))
        transform_functions.append(
            DiscretizeBoundingBox(
                num_x_grid=self.canvas_width, num_y_grid=self.canvas_height
            )
        )
        return transform_functions

    def __call__(self, data):
        _data = self.transform(copy.deepcopy(data))
        return {k: _data[k] for k in self.return_keys}
    


""" Un-used Processors """

class GenTypeProcessor(Processor):
    return_keys = [
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
    ]

    def __init__(
        self,
        index2label: dict,
        canvas_width: int,
        canvas_height: int,
        sort_by_pos: bool = False,
        shuffle_before_sort_by_label: bool = False,
        sort_by_pos_before_sort_by_label: bool = True,
    ):
        super().__init__(
            index2label=index2label,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            sort_by_pos=sort_by_pos,
            shuffle_before_sort_by_label=shuffle_before_sort_by_label,
            sort_by_pos_before_sort_by_label=sort_by_pos_before_sort_by_label,
        )


class GenTypeSizeProcessor(Processor):
    return_keys = [
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
    ]

    def __init__(
        self,
        index2label: dict,
        canvas_width: int,
        canvas_height: int,
        sort_by_pos: bool = False,
        shuffle_before_sort_by_label: bool = True,
        sort_by_pos_before_sort_by_label: bool = False,
    ):
        super().__init__(
            index2label=index2label,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            sort_by_pos=sort_by_pos,
            shuffle_before_sort_by_label=shuffle_before_sort_by_label,
            sort_by_pos_before_sort_by_label=sort_by_pos_before_sort_by_label,
        )


class GenRelationProcessor(Processor):
    return_keys = [
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
        "relations",
    ]

    def __init__(
        self,
        index2label: dict,
        canvas_width: int,
        canvas_height: int,
        sort_by_pos: bool = False,
        shuffle_before_sort_by_label: bool = False,
        sort_by_pos_before_sort_by_label: bool = True,
        relation_constrained_discrete_before_induce_relations: bool = False,
    ):
        super().__init__(
            index2label=index2label,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            sort_by_pos=sort_by_pos,
            shuffle_before_sort_by_label=shuffle_before_sort_by_label,
            sort_by_pos_before_sort_by_label=sort_by_pos_before_sort_by_label,
        )
        self.transform_functions = self.transform_functions[:-1]
        if relation_constrained_discrete_before_induce_relations:
            self.transform_functions.append(
                DiscretizeBoundingBox(
                    num_x_grid=self.canvas_width, num_y_grid=self.canvas_height
                )
            )
            self.transform_functions.append(
                AddCanvasElement(
                    use_discrete=True, discrete_fn=self.transform_functions[-1]
                )
            )
            self.transform_functions.append(AddRelation())
        else:
            self.transform_functions.append(AddCanvasElement())
            self.transform_functions.append(AddRelation())
            self.transform_functions.append(
                DiscretizeBoundingBox(
                    num_x_grid=self.canvas_width, num_y_grid=self.canvas_height
                )
            )
        self.transform = T.Compose(self.transform_functions)


class CompletionProcessor(Processor):
    return_keys = [
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
    ]

    def __init__(
        self,
        index2label: dict,
        canvas_width: int,
        canvas_height: int,
        sort_by_pos: bool = True,
        shuffle_before_sort_by_label: bool = False,
        sort_by_pos_before_sort_by_label: bool = False,
    ):
        super().__init__(
            index2label=index2label,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            sort_by_pos=sort_by_pos,
            shuffle_before_sort_by_label=shuffle_before_sort_by_label,
            sort_by_pos_before_sort_by_label=sort_by_pos_before_sort_by_label,
        )


class ContentAwareProcessor(Processor):
    return_keys = [
        "idx",
        "labels",
        "bboxes",
        "gold_bboxes",
        "content_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
        "discrete_content_bboxes",
    ]

    def __init__(
        self,
        index2label: dict,
        canvas_width: int,
        canvas_height: int,
        metadata: DataFrame,
        sort_by_pos: bool = False,
        shuffle_before_sort_by_label: bool = False,
        sort_by_pos_before_sort_by_label: bool = True,
        filter_threshold: int = 100,
        max_element_numbers: int = 10,
        original_width: float = 513.0,
        original_height: float = 750.0,
    ):
        super().__init__(
            index2label=index2label,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            sort_by_pos=sort_by_pos,
            shuffle_before_sort_by_label=shuffle_before_sort_by_label,
            sort_by_pos_before_sort_by_label=sort_by_pos_before_sort_by_label,
        )
        self.metadata = metadata
        self.max_element_numbers = max_element_numbers
        self.original_width = original_width
        self.original_height = original_height
        self.saliency_map_to_bboxes = SaliencyMapToBBoxes(filter_threshold)
        self.possible_labels: list = []

    def _normalize_bboxes(self, bboxes):
        bboxes = bboxes.float()
        bboxes[:, 0::2] /= self.original_width
        bboxes[:, 1::2] /= self.original_height
        return bboxes

    def __call__(self, filename, idx, split):
        saliency_map = cv2.imread(filename)
        content_bboxes = self.saliency_map_to_bboxes(saliency_map)
        if len(content_bboxes) == 0:
            return None
        content_bboxes = self._normalize_bboxes(content_bboxes)

        if split == "train":
            _metadata = self.metadata[
                self.metadata["poster_path"] == f"train/{idx}.png"
            ][self.metadata["cls_elem"] > 0]
            labels = torch.tensor(list(map(int, _metadata["cls_elem"])))
            bboxes = torch.tensor(list(map(eval, _metadata["box_elem"])))
            if len(labels) == 0:
                return None
            bboxes[:, 2] -= bboxes[:, 0]
            bboxes[:, 3] -= bboxes[:, 1]
            bboxes = self._normalize_bboxes(bboxes)
            if len(labels) <= self.max_element_numbers:
                self.possible_labels.append(labels)

            data = {
                "idx": idx,
                "labels": labels,
                "bboxes": bboxes,
                "content_bboxes": content_bboxes,
            }
        else:
            if len(self.possible_labels) == 0:
                raise RuntimeError("Please process training data first")

            labels = random.choice(self.possible_labels)
            data = {
                "idx": idx,
                "labels": labels,
                "bboxes": torch.zeros((len(labels), 4)),  # dummy
                "content_bboxes": content_bboxes,
            }

        return super().__call__(data)
    


""" Used Processors """

class RefinementProcessor(Processor):
    return_keys = [
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
    ]

    def __init__(
        self,
        index2label: dict,
        canvas_width: int,
        canvas_height: int,
        sort_by_pos: bool = False,
        shuffle_before_sort_by_label: bool = False,
        sort_by_pos_before_sort_by_label: bool = True,
        gaussian_noise_mean: float = 0.0,
        gaussian_noise_std: float = 0.01,
        train_bernoulli_beta: float = 1.0,
    ):
        super().__init__(
            index2label=index2label,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            sort_by_pos=sort_by_pos,
            shuffle_before_sort_by_label=shuffle_before_sort_by_label,
            sort_by_pos_before_sort_by_label=sort_by_pos_before_sort_by_label,
        )
        self.transform_functions = [
            AddGaussianNoise(
                mean=gaussian_noise_mean,
                std=gaussian_noise_std,
                bernoulli_beta=train_bernoulli_beta,
            )
        ] + self.transform_functions
        self.transform = T.Compose(self.transform_functions)


class TextToLayoutProcessor(Processor):
    """
    텍스트를 레이아웃 생성을 위한 데이터로 전처리하는 프로세서 클래스
    
    두 가지 입력을 처리:
    1. 문자열: 사용자 입력 텍스트 ("가나 초콜렛에 대한 홍보물 제작")
    2. 구조화된 데이터: 기존 레이아웃 데이터 (텍스트 + 요소들의 위치/타입 정보)
    """
    
    # 반환할 데이터 키들 정의
    return_keys = [
        "labels",       # UI 요소 타입의 인덱스 (예: [0, 1, 3] = [text, link, title])
        "bboxes",       # 바운딩박스 좌표 (x1, y1, x2, y2)
        "text",         # 정제된 텍스트
        "embedding",    # CLIP 인코더로 생성된 텍스트 임베딩 벡터
    ]

    def __init__(
        self,
        index2label: dict,      # {0: "text", 1: "link", 2: "button", ...} 인덱스->라벨 매핑
        canvas_width: int,      # 타겟 캔버스 너비 (정규화 기준)
        canvas_height: int,     # 타겟 캔버스 높이 (정규화 기준)
    ):
        """
        TextToLayoutProcessor 초기화
        
        Args:
            index2label: 숫자 인덱스를 UI 요소 타입으로 변환하는 딕셔너리
            canvas_width: 표준 캔버스 너비 (모든 레이아웃을 이 크기로 정규화)
            canvas_height: 표준 캔버스 높이 (모든 레이아웃을 이 크기로 정규화)
        """
        self.index2label = index2label  # {0: 'text', 1: 'link', 2: 'button', ...}
        self.label2index = {v: k for k, v in self.index2label.items()}  # {'text': 0, 'link': 1, 'button': 2, ...}
        self.canvas_width = canvas_width    # 표준 캔버스 너비 (예: 400px)
        self.canvas_height = canvas_height  # 표준 캔버스 높이 (예: 600px)
        self.text_encoder = CLIPTextEncoder()  # CLIP 모델을 사용한 텍스트 인코더

    def _scale(self, original_width, elements_):
        """
        레이아웃 요소들을 표준 캔버스 크기에 맞게 스케일링
        
        Args:
            original_width: 원본 캔버스의 너비
            elements_: 스케일링할 요소들의 리스트
            
        Returns:
            스케일링된 요소들의 리스트
            
        Example:
            원본: 800px 캔버스에서 [100, 50, 300, 100] 위치
            타겟: 400px 캔버스 → ratio = 0.5
            결과: [50, 25, 150, 50] 위치로 변환
        """
        elements = copy.deepcopy(elements_)  # 원본 데이터 보존을 위한 깊은 복사
        ratio = self.canvas_width / original_width  # 스케일링 비율 계산
        
        # 모든 요소의 좌표를 비율에 맞게 변환
        for i in range(len(elements)):
            elements[i]["position"][0] = int(ratio * elements[i]["position"][0])  # x 좌표 (좌상단)
            elements[i]["position"][1] = int(ratio * elements[i]["position"][1])  # y 좌표 (좌상단)  
            elements[i]["position"][2] = int(ratio * elements[i]["position"][2])  # width (너비)
            elements[i]["position"][3] = int(ratio * elements[i]["position"][3])  # height (높이)
        return elements

    def __call__(self, data):
        """
        입력 데이터를 처리하여 모델이 사용할 수 있는 형태로 변환
        
        Args:
            data: 문자열 또는 구조화된 데이터 딕셔너리
            
        Returns:
            처리된 데이터 딕셔너리 (text, embedding, labels, bboxes)
        """
        
        # === 케이스 1: 문자열 입력 처리 (사용자 쿼리) ===
        if isinstance(data, str):
            # 사용자가 직접 입력한 자유 텍스트 처리 (예: "가나 초콜렛에 대한 홍보물 제작")
            text = clean_text(data)  # 텍스트 정제 (특수문자, 공백 등 처리)
            
            # CLIP 인코더로 텍스트를 벡터로 변환
            # remove_summary=True: 요약 정보 제거하여 순수 텍스트만 인코딩
            # .to(torch.float16): GPU와 CPU 간 임베딩 차원 호환성을 위한 타입 명시
            embedding = self.text_encoder(clean_text(data, remove_summary=True)).to(torch.float16)
            
            return {
                "text": text,           # 정제된 텍스트
                "embedding": embedding, # CLIP 임베딩 벡터
                # labels, bboxes는 없음 (사용자 쿼리에는 레이아웃 정보가 없음)
            }
        
        # === 케이스 2: 구조화된 데이터 처리 (기존 레이아웃 데이터) ===
        # 텍스트 처리
        text = clean_text(data["text"])  # 레이아웃에 포함된 텍스트 정제
        embedding = self.text_encoder(clean_text(data["text"], remove_summary=True)).to(torch.float16)
        
        # 레이아웃 데이터 추출
        original_width = data["canvas_width"]  # 원본 캔버스 너비
        elements = data["elements"]            # UI 요소들의 리스트
        
        # 캔버스 크기에 맞게 요소들 스케일링
        elements = self._scale(original_width, elements)
        
        # 요소들을 위치 기준으로 정렬 (y좌표 우선, 같으면 x좌표)
        # 위에서 아래로, 왼쪽에서 오른쪽 순으로 정렬
        elements = sorted(elements, key=lambda x: (x["position"][1], x["position"][0]))

        # UI 요소 타입을 숫자 인덱스로 변환
        # 예: ["title", "image", "button"] → [3, 5, 2]
        labels = [self.label2index[element["type"]] for element in elements]
        labels = torch.tensor(labels)  # PyTorch 텐서로 변환

        # 바운딩박스 좌표 추출
        # 각 요소의 position [x, y, width, height] 정보 (x,y는 좌상단 좌표)
        bboxes = [element["position"] for element in elements]
        bboxes = torch.tensor(bboxes)  # PyTorch 텐서로 변환

        return {
            "text": text,                           # 정제된 텍스트
            "embedding": embedding,                 # CLIP 임베딩 벡터
            "labels": labels,                       # UI 요소 타입 인덱스 텐서
            "discrete_gold_bboxes": bboxes,         # 정답 바운딩박스 (훈련용)
            "discrete_bboxes": bboxes,              # 예측용 바운딩박스 (초기값은 정답과 동일)
        }


PROCESSOR_MAP = {
    # Un-Used Processors
    "gent": GenTypeProcessor,
    "gents": GenTypeSizeProcessor,
    "genr": GenRelationProcessor,
    "completion": CompletionProcessor,
    "content": ContentAwareProcessor,

    # Used Processors
    "refinement": RefinementProcessor,
    "text": TextToLayoutProcessor,
}


def create_processor(dataset: str = "webui", task: str = "text", *args, **kwargs):
    processor_cls = PROCESSOR_MAP[task]
    index2label = ID2LABEL[dataset]
    canvas_width, canvas_height = CANVAS_SIZE[dataset]

    processor = processor_cls(
        index2label=index2label,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        *args,
        **kwargs,
    )
    return processor
