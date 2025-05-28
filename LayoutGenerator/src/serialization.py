from .transforms import RelationTypes
from .utilities import CANVAS_SIZE, ID2LABEL, LAYOUT_DOMAIN

PREAMBLE = (
    "Please generate a layout based on the given information. "
    "You need to ensure that the generated layout looks realistic, with elements well aligned and avoiding unnecessary overlap.\n"
    "Task Description: {}\n"
    "Layout Domain: {} layout\n"
    "Canvas Size: canvas width is {}px, canvas height is {}px"
)

HTML_PREFIX = """<html>
<body>
<div class="canvas" style="left: 0px; top: 0px; width: {}px; height: {}px"></div>
"""

HTML_SUFFIX = """</body>
</html>"""

HTML_TEMPLATE = """<div class="{}" style="left: {}px; top: {}px; width: {}px; height: {}px"></div>
"""

HTML_TEMPLATE_WITH_INDEX = """<div class="{}" style="index: {}; left: {}px; top: {}px; width: {}px; height: {}px"></div>
"""


class Serializer:
    """
    레이아웃 데이터를 텍스트 형태로 직렬화하는 기본 클래스
    
    레이아웃 요소들(라벨, 바운딩박스)을 언어 모델이 이해할 수 있는
    시퀀스(sequence) 또는 HTML 형태로 변환합니다.
    서로 다른 작업 유형에 따라 입력과 출력 형식을 유연하게 지원합니다.
    """
    
    def __init__(
        self,
        input_format: str,      # 입력 형식: "seq" (시퀀스) 또는 "html"
        output_format: str,     # 출력 형식: "seq" (시퀀스) 또는 "html"
        index2label: dict,      # 레이블 인덱스 → 레이블 이름 매핑 딕셔너리
        canvas_width: int,      # 캔버스 너비 (픽셀 단위)
        canvas_height: int,     # 캔버스 높이 (픽셀 단위)
        add_index_token: bool = True,   # 요소 인덱스 토큰 추가 여부
        add_sep_token: bool = True,     # 구분자 토큰 추가 여부
        sep_token: str = "|",           # 요소 간 구분자
        add_unk_token: bool = False,    # 미지의 토큰 추가 여부
        unk_token: str = "<unk>",       # 미지의 토큰 문자열
    ):
        """
        Serializer 기본 클래스 초기화
        
        Args:
            input_format: 입력 데이터 형식 ("seq" 또는 "html")
            output_format: 출력 데이터 형식 ("seq" 또는 "html")
            index2label: {0: "title", 1: "image", ...} 형태의 레이블 매핑
            canvas_width: 레이아웃 캔버스의 너비
            canvas_height: 레이아웃 캔버스의 높이
            add_index_token: True면 "title 0", False면 "title"만 출력
            add_sep_token: True면 요소들 사이에 "|" 구분자 추가
            sep_token: 요소 간 구분자 문자 (기본값: "|")
            add_unk_token: 알려지지 않은 위치를 위한 토큰 추가 여부
            unk_token: 미지의 위치를 나타내는 토큰 (기본값: "<unk>")
        """
        # 입출력 형식 설정
        self.input_format = input_format        # "seq" 또는 "html"
        self.output_format = output_format      # "seq" 또는 "html"
        
        # 레이아웃 메타데이터
        self.index2label = index2label          # 인덱스 → 레이블 매핑
        self.canvas_width = canvas_width        # 캔버스 너비
        self.canvas_height = canvas_height      # 캔버스 높이
        
        # 토큰화 옵션
        self.add_index_token = add_index_token  # 인덱스 토큰 추가 여부
        self.add_sep_token = add_sep_token      # 구분자 토큰 추가 여부
        self.sep_token = sep_token              # 구분자 문자
        self.add_unk_token = add_unk_token      # 미지 토큰 추가 여부  
        self.unk_token = unk_token              # 미지 토큰 문자

    def build_input(self, data: dict) -> str:
        """
        입력 데이터를 직렬화된 문자열로 변환
        
        설정된 input_format에 따라 적절한 내부 메서드를 호출합니다.
        각 Serializer 서브클래스에서 구체적인 입력 형식을 정의합니다.
        
        Args:
            data: 레이아웃 데이터 딕셔너리 (labels, bboxes 등 포함)
            
        Returns:
            직렬화된 입력 문자열
            
        Raises:
            ValueError: 지원하지 않는 입력 형식인 경우
        """
        if self.input_format == "seq":
            return self._build_seq_input(data)      # 시퀀스 형태로 변환
        elif self.input_format == "html":
            return self._build_html_input(data)     # HTML 형태로 변환
        else:
            raise ValueError(f"Unsupported input format: {self.input_format}")

    def _build_seq_input(self, data: dict) -> str:
        """
        시퀀스 형태의 입력 데이터 생성 (추상 메서드)
        
        각 서브클래스에서 작업별 입력 형식에 맞게 구현해야 합니다.
        예: "title 0 <unk> <unk> <unk> <unk> | image 1 <unk> <unk> <unk> <unk>"
        """
        raise NotImplementedError

    def _build_html_input(self, data: dict) -> str:
        """
        HTML 형태의 입력 데이터 생성 (추상 메서드)
        
        각 서브클래스에서 작업별 입력 형식에 맞게 구현해야 합니다.
        예: '<div class="title" style="index: 0"></div>'
        """
        raise NotImplementedError

    def build_output(self, data: dict, label_key: str = "labels", bbox_key: str = "discrete_gold_bboxes") -> str:
        """
        출력 데이터를 직렬화된 문자열로 변환
        
        레이아웃의 최종 결과물을 언어 모델이 생성할 수 있는 형태로 변환합니다.
        설정된 output_format에 따라 적절한 내부 메서드를 호출합니다.
        
        Args:
            data: 레이아웃 데이터 딕셔너리
            label_key: 레이블 데이터의 키 이름 (기본값: "labels")
            bbox_key: 바운딩박스 데이터의 키 이름 (기본값: "discrete_gold_bboxes")
            
        Returns:
            직렬화된 출력 문자열
            
        Note:
            출력은 모델이 생성해야 할 최종 레이아웃 형태입니다.
        """
        if self.output_format == "seq":
            return self._build_seq_output(data, label_key, bbox_key)    # 시퀀스 형태
        elif self.output_format == "html":
            return self._build_html_output(data, label_key, bbox_key)   # HTML 형태
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

    def _build_seq_output(self, data: dict, label_key: str, bbox_key: str) -> str:
        """
        시퀀스 형태의 출력 데이터 생성
        
        레이아웃 요소들을 "label index left top width height" 형태의
        시퀀스로 변환합니다. 언어 모델이 생성하기 쉬운 형태입니다.
        
        Args:
            data: 레이아웃 데이터 딕셔너리
            label_key: 레이블 배열의 키 이름
            bbox_key: 바운딩박스 배열의 키 이름
            
        Returns:
            시퀀스 형태의 출력 문자열
            
        Example:
            "title 0 50 10 300 40 | image 1 20 60 200 150"
        """
        labels = data[label_key]    # 레이블 배열 추출
        bboxes = data[bbox_key]     # 바운딩박스 배열 추출
        tokens = []                 # 토큰들을 저장할 리스트

        # 각 레이아웃 요소에 대해 토큰 생성
        for idx in range(len(labels)):
            # 레이블 인덱스를 실제 레이블 이름으로 변환
            label = self.index2label[int(labels[idx])]  # 예: 0 → "title"
            bbox = bboxes[idx].tolist()                 # 텐서를 리스트로 변환
            
            tokens.append(label)        # 레이블 추가 (예: "title")
            
            # 인덱스 토큰 추가 옵션이 활성화된 경우
            if self.add_index_token:
                tokens.append(str(idx)) # 요소 인덱스 추가 (예: "0")
            
            # 바운딩박스 좌표 추가 [left, top, width, height]
            tokens.extend(map(str, bbox))  # 예: ["50", "10", "300", "40"]
            
            # 구분자 토큰 추가 (마지막 요소가 아닌 경우)
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)  # 예: "|"
                
        return " ".join(tokens)  # 모든 토큰을 공백으로 연결

    def _build_html_output(self, data: dict, label_key: str, bbox_key: str) -> str:
        """
        HTML 형태의 출력 데이터 생성
        
        레이아웃 요소들을 HTML div 태그 형태로 변환합니다.
        웹 브라우저에서 직접 시각화할 수 있는 형태입니다.
        
        Args:
            data: 레이아웃 데이터 딕셔너리
            label_key: 레이블 배열의 키 이름
            bbox_key: 바운딩박스 배열의 키 이름
            
        Returns:
            HTML 형태의 출력 문자열
            
        Example:
            '<html><body><div class="canvas" style="..."></div>
             <div class="title" style="index: 0; left: 50px; top: 10px; width: 300px; height: 40px"></div>
             </body></html>'
        """
        labels = data[label_key]    # 레이블 배열 추출
        bboxes = data[bbox_key]     # 바운딩박스 배열 추출
        
        # HTML 문서 시작 부분 (캔버스 정의 포함)
        htmls = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        
        # 인덱스 토큰 추가 여부에 따라 템플릿 선택
        _TEMPLATE = HTML_TEMPLATE_WITH_INDEX if self.add_index_token else HTML_TEMPLATE

        # 각 레이아웃 요소에 대해 HTML div 태그 생성
        for idx in range(len(labels)):
            # 레이블과 바운딩박스 정보 추출
            label = self.index2label[int(labels[idx])]  # 레이블 이름
            bbox = bboxes[idx].tolist()                 # [left, top, width, height]
            
            element = [label]           # 템플릿에 전달할 요소들의 리스트
            
            # 인덱스 토큰 추가 옵션이 활성화된 경우
            if self.add_index_token:
                element.append(str(idx))  # 요소 인덱스 추가
            
            # 바운딩박스 좌표 추가
            element.extend(map(str, bbox))  # [left, top, width, height]
            
            # 선택된 템플릿에 요소들을 적용하여 HTML 태그 생성
            htmls.append(_TEMPLATE.format(*element))
            
        # HTML 문서 종료 부분 추가
        htmls.append(HTML_SUFFIX)
        
        return "".join(htmls)  # 모든 HTML 조각들을 연결



""" Un-Used Serializer Classes """

class GenTypeSerializer(Serializer):
    task_type = "generation conditioned on given element types"
    constraint_type = ["Element Type Constraint: "]
    HTML_TEMPLATE_WITHOUT_ANK = '<div class="{}"></div>\n'
    HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX = '<div class="{}" style="index: {}"></div>\n'

    def _build_seq_input(self, data):
        labels = data["labels"]
        tokens = []

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            tokens.append(label)
            if self.add_index_token:
                tokens.append(str(idx))
            if self.add_unk_token:
                tokens += [self.unk_token] * 4
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)
        return " ".join(tokens)

    def _build_html_input(self, data):
        labels = data["labels"]
        htmls = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        if self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE_WITH_INDEX
        elif self.add_index_token and not self.add_unk_token:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX
        elif not self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE
        else:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            element = [label]
            if self.add_index_token:
                element.append(str(idx))
            if self.add_unk_token:
                element += [self.unk_token] * 4
            htmls.append(_TEMPLATE.format(*element))
        htmls.append(HTML_SUFFIX)
        return "".join(htmls)

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


class GenTypeSizeSerializer(Serializer):
    task_type = "generation conditioned on given element types and sizes"
    constraint_type = ["Element Type and Size Constraint: "]
    HTML_TEMPLATE_WITHOUT_ANK = (
        '<div class="{}" style="width: {}px; height: {}px"></div>\n'
    )
    HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX = (
        '<div class="{}" style="index: {}; width: {}px; height: {}px"></div>\n'
    )

    def _build_seq_input(self, data):
        labels = data["labels"]
        bboxes = data["discrete_gold_bboxes"]
        tokens = []

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            bbox = bboxes[idx].tolist()
            tokens.append(label)
            if self.add_index_token:
                tokens.append(str(idx))
            if self.add_unk_token:
                tokens += [self.unk_token] * 2
            tokens.extend(map(str, bbox[2:]))
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)
        return " ".join(tokens)

    def _build_html_input(self, data):
        labels = data["labels"]
        bboxes = data["discrete_gold_bboxes"]
        htmls = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        if self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE_WITH_INDEX
        elif self.add_index_token and not self.add_unk_token:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX
        elif not self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE
        else:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            bbox = bboxes[idx].tolist()
            element = [label]
            if self.add_index_token:
                element.append(str(idx))
            if self.add_unk_token:
                element += [self.unk_token] * 2
            element.extend(map(str, bbox[2:]))
            htmls.append(_TEMPLATE.format(*element))
        htmls.append(HTML_SUFFIX)
        return "".join(htmls)

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


class GenRelationSerializer(Serializer):
    task_type = (
        "generation conditioned on given element relationships\n"
        "'A left B' means that the center coordinate of A is to the left of the center coordinate of B. "
        "'A right B' means that the center coordinate of A is to the right of the center coordinate of B. "
        "'A top B' means that the center coordinate of A is above the center coordinate of B. "
        "'A bottom B' means that the center coordinate of A is below the center coordinate of B. "
        "'A center B' means that the center coordinate of A and the center coordinate of B are very close. "
        "'A smaller B' means that the area of A is smaller than the ares of B. "
        "'A larger B' means that the area of A is larger than the ares of B. "
        "'A equal B' means that the area of A and the ares of B are very close. "
        "Here, center coordinate = (left + width / 2, top + height / 2), "
        "area = width * height"
    )
    constraint_type = ["Element Type Constraint: ", "Element Relationship Constraint: "]
    HTML_TEMPLATE_WITHOUT_ANK = '<div class="{}"></div>\n'
    HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX = '<div class="{}" style="index: {}"></div>\n'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index2type = RelationTypes.index2type()

    def _build_seq_input(self, data):
        labels = data["labels"]
        relations = data["relations"]
        tokens = []

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            tokens.append(label)
            if self.add_index_token:
                tokens.append(str(idx))
            if self.add_unk_token:
                tokens += [self.unk_token] * 4
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)
        type_cons = " ".join(tokens)
        if len(relations) == 0:
            return self.constraint_type[0] + type_cons
        tokens = []
        for idx in range(len(relations)):
            label_i = relations[idx][2]
            index_i = relations[idx][3]
            if label_i != 0:
                tokens.append("{} {}".format(self.index2label[int(label_i)], index_i))
            else:
                tokens.append("canvas")
            tokens.append(self.index2type[int(relations[idx][4])])
            label_j = relations[idx][0]
            index_j = relations[idx][1]
            if label_j != 0:
                tokens.append("{} {}".format(self.index2label[int(label_j)], index_j))
            else:
                tokens.append("canvas")
            if self.add_sep_token and idx < len(relations) - 1:
                tokens.append(self.sep_token)
        relation_cons = " ".join(tokens)
        return (
            self.constraint_type[0]
            + type_cons
            + "\n"
            + self.constraint_type[1]
            + relation_cons
        )

    def _build_html_input(self, data):
        labels = data["labels"]
        relations = data["relations"]
        htmls = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        if self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE_WITH_INDEX
        elif self.add_index_token and not self.add_unk_token:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX
        elif not self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE
        else:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            element = [label]
            if self.add_index_token:
                element.append(str(idx))
            if self.add_unk_token:
                element += [self.unk_token] * 4
            htmls.append(_TEMPLATE.format(*element))
        htmls.append(HTML_SUFFIX)
        type_cons = "".join(htmls)
        if len(relations) == 0:
            return self.constraint_type[0] + type_cons
        tokens = []
        for idx in range(len(relations)):
            label_i = relations[idx][2]
            index_i = relations[idx][3]
            if label_i != 0:
                tokens.append("{} {}".format(self.index2label[int(label_i)], index_i))
            else:
                tokens.append("canvas")
            tokens.append(self.index2type[int(relations[idx][4])])
            label_j = relations[idx][0]
            index_j = relations[idx][1]
            if label_j != 0:
                tokens.append("{} {}".format(self.index2label[int(label_j)], index_j))
            else:
                tokens.append("canvas")
            if self.add_sep_token and idx < len(relations) - 1:
                tokens.append(self.sep_token)
        relation_cons = " ".join(tokens)
        return (
            self.constraint_type[0]
            + type_cons
            + "\n"
            + self.constraint_type[1]
            + relation_cons
        )


class CompletionSerializer(Serializer):
    task_type = "layout completion"
    constraint_type = ["Partial Layout: "]

    def _build_seq_input(self, data):
        data["partial_labels"] = data["labels"][:1]
        data["partial_bboxes"] = data["discrete_bboxes"][:1, :]
        return self._build_seq_output(data, "partial_labels", "partial_bboxes")

    def _build_html_input(self, data):
        data["partial_labels"] = data["labels"][:1]
        data["partial_bboxes"] = data["discrete_bboxes"][:1, :]
        return self._build_html_output(data, "partial_labels", "partial_bboxes")

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


class ContentAwareSerializer(Serializer):
    task_type = (
        "content-aware layout generation\n"
        "Please place the following elements to avoid salient content, and underlay must be the background of text or logo."
    )
    constraint_type = ["Content Constraint: ", "Element Type Constraint: "]
    CONTENT_TEMPLATE = "left {}px, top {}px, width {}px, height {}px"

    def _build_seq_input(self, data):
        labels = data["labels"]
        content_bboxes = data["discrete_content_bboxes"]

        tokens = []
        for idx in range(len(content_bboxes)):
            content_bbox = content_bboxes[idx].tolist()
            tokens.append(self.CONTENT_TEMPLATE.format(*content_bbox))
            if self.add_index_token and idx < len(content_bboxes) - 1:
                tokens.append(self.sep_token)
        content_cons = " ".join(tokens)

        tokens = []
        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            tokens.append(label)
            if self.add_index_token:
                tokens.append(str(idx))
            if self.add_unk_token:
                tokens += [self.unk_token] * 4
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)
        type_cons = " ".join(tokens)
        return (
            self.constraint_type[0]
            + content_cons
            + "\n"
            + self.constraint_type[1]
            + type_cons
        )
    


""" Used Serializer Classes """

class RefinementSerializer(Serializer):
    task_type = "layout refinement"
    constraint_type = ["Noise Layout: "]

    def _build_seq_input(self, data):
        return self._build_seq_output(data, "labels", "discrete_bboxes")

    def _build_html_input(self, data):
        return self._build_html_output(data, "labels", "discrete_bboxes")

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


class TextToLayoutSerializer(Serializer):
    task_type = (
        "text-to-layout\n"
        "There are ten optional element types, including: image, icon, logo, background, title, description, text, link, input, button. "
        "Please do not exceed the boundaries of the canvas. "
        "Besides, do not generate elements at the edge of the canvas, that is, reduce top: 0px and left: 0px predictions as much as possible."
    )
    constraint_type = ["Text: "]

    def _build_seq_input(self, data):
        return data["text"]

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


SERIALIZER_MAP = {
    # Un-used serializers
    "gent": GenTypeSerializer,
    "gents": GenTypeSizeSerializer,
    "genr": GenRelationSerializer,
    "completion": CompletionSerializer,
    "content": ContentAwareSerializer,

    # Used serializers
    "refinement": RefinementSerializer,
    "text": TextToLayoutSerializer,
}


def create_serializer(
    dataset: str,          # 데이터셋 이름 (예: "publaynet", "rico" 등)
    task: str,             # 작업 유형 (예: "refinement", "text" 등)
    input_format: str,     # 입력 형식: "seq" (시퀀스) 또는 "html"
    output_format: str,    # 출력 형식: "seq" (시퀀스) 또는 "html"
    add_index_token: bool, # 요소 인덱스 토큰 추가 여부 (bool)
    add_sep_token: bool,   # 구분자 토큰 추가 여부 (bool)
    add_unk_token: bool,   # 미지의 토큰 추가 여부 (bool)
):
    """
    작업 유형과 데이터셋에 맞는 Serializer 인스턴스를 생성하는 팩토리 함수
    
    이 함수는 다양한 레이아웃 작업(텍스트→레이아웃, 레이아웃 정제 등)에 대해
    적절한 Serializer 클래스를 선택하고, 데이터셋별 메타데이터를 자동으로 설정하여
    완전히 구성된 Serializer 객체를 반환합니다.
    
    Args:
        dataset: 데이터셋 이름 ("publaynet", "rico", "magazine" 등)
                 - ID2LABEL과 CANVAS_SIZE 딕셔너리의 키로 사용됨
        task: 수행할 작업 유형 ("refinement", "text", "gent" 등)
              - SERIALIZER_MAP의 키로 적절한 Serializer 클래스 선택
        input_format: 입력 데이터 형식 ("seq" 또는 "html")
        output_format: 출력 데이터 형식 ("seq" 또는 "html")
        add_index_token: True시 "title 0", False시 "title"만 출력
        add_sep_token: True시 요소들 사이에 "|" 구분자 추가
        add_unk_token: True시 알려지지 않은 위치에 "<unk>" 토큰 사용
        
    Returns:
        완전히 구성된 Serializer 서브클래스 인스턴스
        
    Example:
        >>> serializer = create_serializer(
        ...     dataset="publaynet",
        ...     task="refinement", 
        ...     input_format="seq",
        ...     output_format="html",
        ...     add_index_token=True,
        ...     add_sep_token=True,
        ...     add_unk_token=False
        ... )
        >>> # RefinementSerializer 인스턴스가 publaynet 설정으로 생성됨
    """
    # 1. 작업 유형에 따른 Serializer 클래스 선택
    # SERIALIZER_MAP: {"refinement": RefinementSerializer, "text": TextToLayoutSerializer, ...}
    serializer_cls = SERIALIZER_MAP[task]
    
    # 2. 데이터셋별 메타데이터 조회
    # 레이블 인덱스 → 레이블 이름 매핑 딕셔너리 가져오기
    # 예: {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
    index2label = ID2LABEL[dataset]
    
    # 데이터셋별 캔버스 크기 가져오기 (픽셀 단위)
    # 예: publaynet의 경우 (120, 160), rico의 경우 (360, 640)
    canvas_width, canvas_height = CANVAS_SIZE[dataset]
    
    # 3. 선택된 Serializer 클래스 인스턴스 생성
    # 모든 필요한 설정을 전달하여 완전히 구성된 객체 생성
    serializer = serializer_cls(
        input_format=input_format,          # 입력 데이터 직렬화 형식
        output_format=output_format,        # 출력 데이터 직렬화 형식
        index2label=index2label,            # 레이블 매핑 정보
        canvas_width=canvas_width,          # 캔버스 너비
        canvas_height=canvas_height,        # 캔버스 높이
        add_index_token=add_index_token,    # 인덱스 토큰 옵션
        add_sep_token=add_sep_token,        # 구분자 토큰 옵션
        add_unk_token=add_unk_token,        # 미지 토큰 옵션
    )
    
    # 4. 완전히 구성된 Serializer 인스턴스 반환
    return serializer


from typing import Any

def build_prompt(
    serializer: Any,  # Serializer 서브클래스들이 task_type을 가지므로 Any 사용
    exemplars: list[dict],
    test_data: dict,
    dataset: str,
    max_length: int = 8000,
    separator_in_samples: str = "\n",
    separator_between_samples: str = "\n\n",
) -> str:
    """
    Few-shot learning을 위한 완성된 프롬프트를 구성합니다.
    
    선택된 예시들과 테스트 데이터를 조합하여 언어 모델이 이해할 수 있는
    구조화된 프롬프트를 생성합니다.
    
    Args:
        serializer: 데이터를 텍스트 형태로 직렬화하는 Serializer 객체
        exemplars: ExemplarSelection으로 선택된 예시 데이터들의 리스트
                  각 요소는 labels, bboxes 등을 포함하는 딕셔너리
        test_data: 레이아웃을 생성할 대상 테스트 데이터 딕셔너리
        dataset: 데이터셋 이름 (canvas 크기, 도메인 정보 조회용)
        max_length: 프롬프트 최대 길이 제한 (토큰 수 아닌 문자열 길이)
        separator_in_samples: 각 예시 내에서 입력과 출력을 구분하는 구분자
        separator_between_samples: 예시들 간의 구분자
        
    Returns:
        완성된 Few-shot 프롬프트 문자열
        
    프롬프트 구조:
        PREAMBLE (작업 설명, 도메인, 캔버스 크기)
        ↓
        예시 1: 입력 + separator_in_samples + 출력
        ↓ separator_between_samples
        예시 2: 입력 + separator_in_samples + 출력
        ↓ separator_between_samples
        ...
        ↓ separator_between_samples
        테스트 데이터: 입력 + separator_in_samples (출력은 모델이 생성)
    """
    # 프롬프트 시작 부분: 작업 설명과 메타데이터 구성
    prompt = [
        PREAMBLE.format(
            serializer.task_type,           # 작업 유형 (예: "text-to-layout")
            LAYOUT_DOMAIN[dataset],         # 레이아웃 도메인 (예: "web UI layout")
            *CANVAS_SIZE[dataset]           # 캔버스 크기 [width, height] 언패킹
        )
    ]
    
    # Few-shot 예시들을 순차적으로 프롬프트에 추가
    for i in range(len(exemplars)):
        # 현재 예시의 입력-출력 쌍 구성
        _prompt = (
            serializer.build_input(exemplars[i])    # 예시의 입력 부분 직렬화
            + separator_in_samples                   # 입력과 출력 사이 구분자 ("\n")
            + serializer.build_output(exemplars[i]) # 예시의 출력 부분 직렬화
        )
        
        # 길이 제한 체크: 새로운 예시를 추가했을 때 최대 길이를 초과하는지 확인
        # separator_between_samples.join(prompt): 현재까지의 프롬프트를 연결
        # + _prompt: 새로운 예시 추가
        if len(separator_between_samples.join(prompt) + _prompt) <= max_length:
            prompt.append(_prompt)  # 제한 내라면 예시 추가
        else:
            break  # 길이 초과 시 더 이상 예시 추가하지 않음
    
    # 테스트 데이터의 입력 부분만 추가 (출력은 모델이 생성해야 함)
    prompt.append(
        serializer.build_input(test_data)   # 테스트 데이터 입력 직렬화
        + separator_in_samples              # 입력 후 구분자 (출력 위치 표시)
    )
    
    # 모든 프롬프트 섹션들을 구분자로 연결하여 최종 프롬프트 생성
    return separator_between_samples.join(prompt)  # "\n\n"로 섹션들 연결


if __name__ == "__main__":
    import torch

    from utilities import ID2LABEL

    ls = RefinementSerializer(
        input_format="seq",
        output_format="html",
        index2label=ID2LABEL["publaynet"],
        canvas_width=120,
        canvas_height=160,
        add_sep_token=True,
        add_unk_token=False,
        add_index_token=True,
    )
    labels = torch.tensor([4, 4, 1, 1, 1, 1])
    bboxes = torch.tensor(
        [
            [29, 14, 59, 2],
            [10, 18, 99, 57],
            [10, 79, 99, 4],
            [10, 85, 99, 7],
            [10, 99, 47, 50],
            [61, 99, 47, 50],
        ]
    )

    rearranged_labels = torch.tensor([1, 4, 1, 4, 1, 1])
    relations = torch.tensor([[4, 1, 0, 1, 4], [1, 2, 1, 3, 2]])
    data = {
        "labels": labels,
        "discrete_bboxes": bboxes,
        "discrete_gold_bboxes": bboxes,
        "relations": relations,
        "rearranged_labels": rearranged_labels,
    }
    print("--------")
    print(ls.build_input(data))
    print("--------")
    print(ls.build_output(data))