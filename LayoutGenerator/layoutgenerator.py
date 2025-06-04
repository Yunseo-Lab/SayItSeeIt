"""
LayoutGenerator: Text-to-Layout 파이프라인

사용자의 자연어 입력을 받아 UI 레이아웃을 자동 생성하는 시스템
"""
import os
import sys
import torch
import traceback
from typing import List, Dict, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# 현재 경로를 기준으로 src 모듈을 임포트할 수 있도록 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.preprocess import create_processor
from src.utilities import ID2LABEL, RAW_DATA_PATH, read_pt, write_pt, read_json
from src.selection import create_selector
from src.serialization import create_serializer, build_prompt
from src.parsing import Parser
from src.ranker import Ranker
from src.visualization import Visualizer
from src.generators.layout_generator import generate_layouts


# 상수 정의
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1200
DEFAULT_NUM_RETURN = 2
DEFAULT_NUM_PROMPT = 5

class TextToLayoutPipeline:
    """
    텍스트에서 레이아웃을 생성하는 메인 파이프라인 클래스
    
    자연어 설명을 입력받아 UI 레이아웃을 자동 생성하는 전체 워크플로우를 관리합니다.
    """
    
    def __init__(
        self,
        dataset: str = "webui",
        task: str = "text",
        input_format: str = "seq",
        output_format: str = "html",
        add_unk_token: bool = False,
        add_index_token: bool = False,
        add_sep_token: bool = True,
        candidate_size: int = -1,
        num_prompt: int = DEFAULT_NUM_PROMPT,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = 1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        num_return: int = DEFAULT_NUM_RETURN,
        stop_token: str = "\n\n",
    ):
        """
        파이프라인 초기화
        
        Args:
            dataset: 데이터셋 이름 ("webui", "rico", "publaynet", "posterlayout")
            task: 작업 유형 ("text", "refinement" 등)
            input_format: 입력 형식 ("seq", "html")
            output_format: 출력 형식 ("seq", "html")
            num_prompt: Few-shot learning에 사용할 예시 개수
            model: 사용할 언어 모델명
        """
        load_dotenv()
        
        # 파라미터 검증
        if dataset not in ID2LABEL:
            raise ValueError(f"지원하지 않는 데이터셋: {dataset}")
            
        self.dataset = dataset
        self.task = task
        self.input_format = input_format
        self.output_format = output_format
        self.add_unk_token = add_unk_token
        self.add_index_token = add_index_token
        self.add_sep_token = add_sep_token
        self.candidate_size = candidate_size
        self.num_prompt = num_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.num_return = num_return
        self.stop_token = stop_token

        # 컴포넌트 초기화
        self.processor = create_processor(dataset, task)
        self.serializer = create_serializer(
            dataset, task, input_format, output_format,
            add_index_token, add_sep_token, add_unk_token
        )
        self.parser = Parser(dataset=dataset, output_format=output_format)
        self.ranker = Ranker()

    def get_processed_data(self, split: str) -> List[Dict]:
        """
        데이터셋을 전처리하고 캐시된 결과를 반환
        
        Args:
            split: 데이터 분할 ("train", "val", "test")
            
        Returns:
            전처리된 데이터 리스트
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"지원하지 않는 split: {split}")
            
        base_dir = os.path.dirname(os.path.abspath(__file__)) 
        filename = os.path.join(
            base_dir, "dataset", self.dataset, "processed", self.task, f"{split}.pt"
        )
        raw_path = os.path.join(RAW_DATA_PATH(self.dataset), f"{split}.json")
        
        # JSON 파일이 존재하는지 확인
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"원본 데이터 파일을 찾을 수 없습니다: {raw_path}")
        
        # 캐시된 파일이 있고, JSON 파일보다 최신이면 로드
        if os.path.exists(filename):
            cache_mtime = os.path.getmtime(filename)
            json_mtime = os.path.getmtime(raw_path)
            
            if cache_mtime >= json_mtime:
                print(f"캐시된 {split} 데이터 로드 중...")
                return read_pt(filename, map_location="cpu")
            else:
                print(f"JSON 파일이 변경되어 {split} 데이터를 다시 전처리합니다...")
        
        # 원본 데이터 전처리
        data = []
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        raw_data = read_json(raw_path)
        
        for rd in tqdm(raw_data, desc=f"{split} 데이터 전처리 중..."):
            data.append(self.processor(rd))
            
        write_pt(filename, data)
        print(f"{split} 데이터 전처리 완료 및 캐시 저장: {filename}")
        return data

    def _select_exemplars(self, train_data: List[Dict], test_item: Dict) -> List[Dict]:
        """Few-shot learning을 위한 예시 선택"""
        selector = create_selector(
            task=self.task,
            train_data=train_data,
            candidate_size=self.candidate_size,
            num_prompt=self.num_prompt
        )
        result = selector(test_item)
        return result if result is not None else []

    def _build_prompt(self, exemplars: List[Dict], test_item: Dict, num_images: int) -> str:
        """Few-shot 프롬프트 생성"""
        return build_prompt(
            self.serializer, exemplars, test_item, self.dataset, num_images=num_images
        ) 
    
    def generate_layouts(self, prompt, n: int = DEFAULT_NUM_RETURN) -> List[str]:
        return generate_layouts(prompt, n)

    def parse_response(self, response):
        return self.parser(response)

    def rank_layouts(self, parsed):
        return self.ranker(parsed)
    
    def map_labels_to_bboxes(self, ranked: List) -> List[Dict]:
        """
        레이블과 바운딩박스를 매핑하여 딕셔너리 형태로 변환
        
        Args:
            ranked: 랭킹된 레이아웃 리스트 [(labels, bboxes), ...]
            use_pixels: True면 픽셀 단위로 변환, False면 정규화된 좌표 유지
            
        Returns:
            각 레이아웃의 요소별 위치 정보를 담은 딕셔너리 리스트
        """
        if not ranked:
            return []
            
        ranked_with_contents = []
        
        # 픽셀 단위 변환을 위한 캔버스 크기 가져오기
        from src.utilities import CANVAS_SIZE
        canvas_width, canvas_height = CANVAS_SIZE[self.dataset]
        
        for item in ranked:
            labels, bboxes = item
            layout_dict = {}
            label_counts = {}  # 각 라벨의 출현 횟수를 추적
            
            for i, (label, bbox) in enumerate(zip(labels, bboxes)):
                label_name = ID2LABEL[self.dataset].get(label.item(), str(label.item()))
                
                # 같은 라벨이 이미 존재하는지 확인하여 인덱스 추가
                if label_name in label_counts:
                    label_counts[label_name] += 1
                    unique_label_name = f"{label_name}_{label_counts[label_name]}"
                else:
                    label_counts[label_name] = 1
                    unique_label_name = label_name
                
                # 정규화된 좌표를 픽셀 좌표로 변환
                x, y, w, h = bbox.tolist()
                layout_dict[unique_label_name] = [
                    round(x * canvas_width),  # x 좌표 (픽셀)
                    round(y * canvas_height), # y 좌표 (픽셀)
                    round(w * canvas_width),  # width (픽셀)
                    round(h * canvas_height)  # height (픽셀)
                ]


            ranked_with_contents.append(layout_dict)

        return ranked_with_contents

    def run(self, user_text: str = "", num_images: int = 0) -> List[Dict]:
        """
        텍스트로부터 레이아웃을 생성하는 전체 파이프라인 실행
        
        Args:
            user_text: 사용자 입력 텍스트
            use_pixels: True면 픽셀 단위로 출력, False면 정규화된 좌표 출력
            
        Returns:
            생성된 레이아웃들의 요소별 위치 정보
            
        Raises:
            ValueError: 입력값이 유효하지 않을 때
            RuntimeError: 파이프라인 실행 중 오류 발생 시
        """
        if not user_text.strip():
            raise ValueError("사용자 텍스트가 비어있습니다.")
        
        try:
            # Detect device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"사용 중인 디바이스: {device}")

            # 1. 훈련 데이터 로드
            train = self.get_processed_data("train")
            print(f"훈련 데이터 로드 완료: {len(train)}개 샘플")

            # 2. 사용자 입력 전처리
            test = self.processor(user_text)

            # 3. 예시 선택
            exemplars = self._select_exemplars(train, test)
            print(f"선택된 예시 수: {len(exemplars)}개")

            # 4. 프롬프트 생성
            prompt = self._build_prompt(exemplars, test, num_images)

            # 5. 레이아웃 생성
            response = self.generate_layouts(prompt, self.num_return)

            # 6. 응답 파싱
            parsed = self.parse_response(response)
            if not parsed:
                print("경고: 파싱된 레이아웃이 없습니다.")
                return []
            
            # 7. 레이아웃 랭킹
            ranked = self.rank_layouts(parsed)
            
            return ranked
            
        except Exception as e:
            traceback.print_exc()  # 전체 스택 트레이스 출력
            raise RuntimeError(f"파이프라인 실행 중 오류 발생: {str(e)}") from e


def main():
    """메인 실행 함수"""
    try:
        # 파이프라인 초기화
        dataset="cardnews"
        pipeline = TextToLayoutPipeline(dataset=dataset)
        visualizer = Visualizer(dataset=dataset)
        
        # 테스트 텍스트
        user_text = "왼쪽 위에 제목이 있고, 제목 아래에는 본문이 있고, 오른쪽에는 이미지가 있습니다. 아래에는 버튼이 있습니다."
        print(f"입력 텍스트: {user_text}")
        print("-" * 50)
        

        # 레이아웃 생성 (픽셀 단위)
        results = pipeline.run(user_text=user_text)

        # 시각화 
        visualizer.visualize(results)
        
        # 레이아웃 요소별 위치 정보 매핑
        results = pipeline.map_labels_to_bboxes(results)
        print("\n=== 생성된 레이아웃 (픽셀 단위) ===")
        for i, layout in enumerate(results, 1):
            print(f"\n레이아웃 {i}:")
            for element, coords in layout.items():
                x, y, w, h = coords
                print(f"  {element}: x={x}px, y={y}px, width={w}px, height={h}px")
                
    except Exception as e:
        print(f"오류 발생: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
