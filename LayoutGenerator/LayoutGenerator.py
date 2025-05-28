import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

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
from src.visualization import Visualizer, create_image_grid
from src.generator import generate_layout

class TextToLayoutPipeline:
    def __init__(
        self,
        dataset="webui",
        task="text",
        input_format="seq",
        output_format="html",
        add_unk_token=False,
        add_index_token=False,
        add_sep_token=True,
        candidate_size=-1,
        num_prompt=10,
        model="gpt-4.1-mini",
        temperature=0.3,
        max_tokens=1200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        num_return=10,
        stop_token="\n\n",
    ):
        load_dotenv()
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

        self.processor = create_processor(dataset, task)
        self.serializer = create_serializer(
            dataset, task, input_format, output_format,
            add_index_token, add_sep_token, add_unk_token
        )
        self.parser = Parser(dataset=dataset, output_format=output_format)
        self.ranker = Ranker()
        self.visualizer = Visualizer(dataset)
        self.client = OpenAI()

    def get_processed_data(self, split):
        base_dir = os.path.dirname(os.path.abspath(__file__)) 
        filename = os.path.join(
            base_dir, "dataset", self.dataset, "processed", self.task, f"{split}.pt"
        )
        if os.path.exists(filename):
            return read_pt(filename, map_location="cpu")
        data = []
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        raw_path = os.path.join(RAW_DATA_PATH(self.dataset), f"{split}.json")
        raw_data = read_json(raw_path)
        for rd in tqdm(raw_data, desc=f"{split} data processing..."):
            data.append(self.processor(rd))
        write_pt(filename, data)
        return data

    def select_exemplars(self, train_data, test_item):
        selector = create_selector(
            task=self.task,
            train_data=train_data,
            candidate_size=self.candidate_size,
            num_prompt=self.num_prompt
        )
        return selector(test_item)

    def build_prompt(self, exemplars, test_item):
        return build_prompt(
            self.serializer, exemplars, test_item, self.dataset
        ) 
    
    def generate_layout(self, prompt):
        return generate_layout(prompt)

    def parse_response(self, response):
        return self.parser(response)

    def rank_layouts(self, parsed):
        return self.ranker(parsed)
    
    def map_labels_to_bboxes(self, ranked, use_pixels=False):
        """
        레이블과 바운딩박스를 매핑하여 딕셔너리 형태로 변환
        
        Args:
            ranked: 랭킹된 레이아웃 리스트 [(labels, bboxes), ...]
            use_pixels (bool): True면 픽셀 단위로 변환, False면 정규화된 좌표 유지
            
        Returns:
            List[Dict]: 각 레이아웃의 요소별 위치 정보를 담은 딕셔너리 리스트
        """
        ranked_with_contents = []
        
        # 캔버스 크기 가져오기 (픽셀 단위 변환용)
        if use_pixels:
            from src.utilities import CANVAS_SIZE
            canvas_width, canvas_height = CANVAS_SIZE[self.dataset]
        
        for item in ranked:
            labels, bboxes = item

            # 레이블 이름과 해당 바운딩 박스를 맵핑
            layout_dict = {}
            for i, (label, bbox) in enumerate(zip(labels, bboxes)):
                label_name = ID2LABEL[self.dataset].get(label.item(), str(label.item()))
                
                if use_pixels:
                    # 정규화된 좌표를 픽셀 좌표로 변환
                    x, y, w, h = bbox.tolist()
                    layout_dict[label_name] = [
                        round(x * canvas_width),  # x 좌표 (픽셀)
                        round(y * canvas_height), # y 좌표 (픽셀)
                        round(w * canvas_width),  # width (픽셀)
                        round(h * canvas_height)  # height (픽셀)
                    ]
                else:
                    # 정규화된 좌표 그대로 사용
                    layout_dict[label_name] = bbox.tolist()

            ranked_with_contents.append(layout_dict)

        return ranked_with_contents

    def visualize(self, ranked):
        images = self.visualizer(ranked)
        grid_img = create_image_grid(images)
        # Create output directory and save path
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "output_poster.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        grid_img.save(output_path)

    def run(self, user_text="", use_pixels=False):
        """
        텍스트로부터 레이아웃을 생성하는 전체 파이프라인 실행
        
        Args:
            user_text (str): 사용자 입력 텍스트
            use_pixels (bool): True면 픽셀 단위로 출력, False면 정규화된 좌표 출력
            
        Returns:
            List[Dict]: 생성된 레이아웃들의 요소별 위치 정보
        """
        train = self.get_processed_data("train")

        test = self.processor(user_text)

        exemplars = self.select_exemplars(train, test)

        prompt = self.build_prompt(exemplars, test)

        response = self.generate_layout(prompt)
        
        parsed = self.parse_response(response)
        
        ranked = self.rank_layouts(parsed)

        self.visualize(ranked)

        formatted_ranked = self.map_labels_to_bboxes(ranked, use_pixels=use_pixels)

        return formatted_ranked


if __name__ == "__main__":
    pipeline = TextToLayoutPipeline()

    user_text = "가나 초콜렛에 대한 홍보물 제작"
    
    # # 정규화된 좌표로 출력 (기본값)
    # result_normalized = pipeline.run(user_text=user_text, use_pixels=False)
    # print("=== 정규화된 좌표 (0.0~1.0) ===")
    # print(result_normalized)
    
    # 픽셀 단위로 출력
    result_pixels = pipeline.run(user_text=user_text, use_pixels=True)
    print("\n=== 픽셀 단위 좌표 ===")
    print(result_pixels)