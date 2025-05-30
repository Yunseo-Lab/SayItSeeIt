from layoutgenerator import TextToLayoutPipeline
from src.generators.copy_generator import generate_copies

pipeline = TextToLayoutPipeline(dataset="cardnews")

user_text = "왼쪽 위에 제목이 있고, 제목 아래에는 본문이 있고, 오른쪽에는 이미지가 있습니다. 아래에는 버튼이 있습니다."

results = pipeline.run(user_text=user_text)

keys_per_sample = [list(sample.keys()) for sample in pipeline.map_labels_to_bboxes(results)]

copy_result = generate_copies(keys_per_sample)

# 텍스트 내용과 함께 시각화
pipeline.visualize(results, copy=copy_result)

print("Generated Layouts with contents:", copy_result)

