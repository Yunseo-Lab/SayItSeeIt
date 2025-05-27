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
from src.contents import generate_contents


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
        # base_dir = os.path.dirname(os.getcwd())
        base_dir = os.path.dirname(os.path.abspath(__file__))  # main.py 위치 기준
        filename = os.path.join(
            base_dir, "dataset", self.dataset, "processed", self.task, f"{split}.pt"
        )
        if os.path.exists(filename):
            return read_pt(filename, map_location="cpu")
        data = []
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        raw_path = os.path.join(RAW_DATA_PATH(self.dataset), f"{split}.json")
        # raw_path = os.path.join(base_dir, "dataset", self.dataset, f"raw/{split}.json")
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

    def call_model(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            n=self.num_return,
        )
        return response

    def parse_response(self, response):
        return self.parser(response)

    def rank_layouts(self, parsed):
        return self.ranker(parsed)
    
    def map_labels_to_bboxes(self, ranked):
        ranked_with_contents = []
        for item in ranked:
            labels, bboxes = item

            # 레이블 이름과 해당 바운딩 박스를 맵핑
            layout_dict = {}
            for i, (label, bbox) in enumerate(zip(labels, bboxes)):
                label_name = ID2LABEL[self.dataset].get(label.item(), str(label.item()))
                layout_dict[label_name] = bbox.tolist()

            ranked_with_contents.append(layout_dict)

        return ranked_with_contents

    def visualize(self, ranked_with_contents):
        images = self.visualizer(ranked_with_contents)
        grid_img = create_image_grid(images)
        # Create output directory and save path
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "output_poster.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        grid_img.save(output_path)

    def run(self, test_idx=0, user_text=""):
        train = self.get_processed_data("train")
        # _ = self.get_processed_data("val")
        # test = self.get_processed_data("test")
        
        test = [self.processor(user_text)]

        exemplars = self.select_exemplars(train, test[test_idx])
        prompt = self.build_prompt(exemplars, test[test_idx])
        response = self.call_model(prompt)
        print(f"Response: {response.choices[0].message.content}")
        parsed = self.parse_response(response)
        ranked = self.rank_layouts(parsed)
        ranked_with_contents = self.map_labels_to_bboxes(ranked)

        # visualize엔 content 포함된 리스트 전달
        self.visualize(ranked_with_contents)

        return ranked_with_contents[0]


if __name__ == "__main__":
    pipeline = TextToLayoutPipeline(dataset="webui")

    user_text = "가나 초콜렛에 대한 홍보물 제작"
    result = pipeline.run(user_text=user_text)
    print(result)