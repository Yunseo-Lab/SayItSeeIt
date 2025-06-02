# filepath: /Users/localgroup/Documents/workspace/SayItSeeIt/LayoutGenerator/app.py

from main import TextToLayoutPipeline, GraphState, initialize_graph
from src.generators.copy_generator import generate_copies
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import invoke_graph, random_uuid
import gradio as gr
import os

# 파이프라인 초기화
dataset = "cardnews"  # 사용할 데이터셋 이름
pipeline = TextToLayoutPipeline(dataset=dataset)

def run_layout_generation(query_text, image_files):
    """Gradio에서 호출할 함수 - 버튼 누른 순간부터 이미지 변경 감지"""
    import threading
    from io import BytesIO
    from PIL import Image
    import time
    import hashlib
    # 이미지 파일명 리스트 생성
    image_filenames = []
    if image_files:
        for file in image_files:
            filename = os.path.basename(file.name)
            image_filenames.append(filename)
    # 초기 상태 정의
    initial_state: GraphState = {
        "pipeline": pipeline,
        "query": query_text,
        "layout": [],
        "copy": [],
        "images": image_filenames,
    }
    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})
    app = initialize_graph()
    # invoke_graph를 백그라운드에서 실행
    def run_graph():
        try:
            invoke_graph(app, initial_state, config)
        except Exception:
            pass
    t = threading.Thread(target=run_graph)
    t.start()
    # 이미지 변경 감지 루프
    output_image_path = "output/output_poster.png"
    yielded_hashes = set()
    changed_count = 0
    max_changes = 2  # 두 번 바뀔 때까지 반환
    max_wait = 20  # 최대 20초 대기
    waited = 0
    prev_img_hash = None
    # 최초 진입 시 이전 작업 결과(캐시) 무시: 첫 해시를 prev_img_hash로 저장하고, 그 다음부터 새로운 해시만 yield
    first_hash_set = False
    while changed_count < max_changes and waited < max_wait:
        try:
            if os.path.exists(output_image_path):
                with open(output_image_path, "rb") as f:
                    img_bytes = f.read()
                img_hash = hashlib.md5(img_bytes).hexdigest()
                if not first_hash_set:
                    prev_img_hash = img_hash
                    first_hash_set = True
                    time.sleep(0.1)
                    waited += 0.1
                    continue
                # 이전 작업 결과와 다를 때만 yield
                if img_hash not in yielded_hashes and img_hash != prev_img_hash:
                    from PIL import Image
                    from io import BytesIO
                    img = Image.open(BytesIO(img_bytes)).copy()
                    yield img
                    yielded_hashes.add(img_hash)
                    changed_count += 1
                else:
                    time.sleep(0.1)
                    waited += 0.1
                    continue
            else:
                if changed_count == 0:
                    yield None
            time.sleep(0.1)
            waited += 0.1
        except Exception:
            pass
    # 만약 한 번만 바뀌고 끝났으면 마지막 이미지를 한 번 더 출력 (Gradio에서 마지막 상태 보장)
    if changed_count == 1 and os.path.exists(output_image_path):
        try:
            with open(output_image_path, "rb") as f:
                img_bytes = f.read()
            from PIL import Image
            from io import BytesIO
            img = Image.open(BytesIO(img_bytes)).copy()
            yield img
        except Exception:
            pass

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    with gr.Blocks(title="카드뉴스 생성기") as demo:
        gr.Markdown("# 🎨 카드뉴스 생성기")
        gr.Markdown("원하는 카드뉴스 레이아웃을 설명하고 이미지를 업로드하세요!")
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="레이아웃 요청사항",
                    placeholder="예: 빙그레 초코우유에 대한 카드뉴스를 제작할거야. 제목은 '초코 타임!'이야...",
                    lines=4,
                    value="빙그레 초코 우유에 대한 홍보 카드뉴스를 만들거야. 왼쪽 면을 거의 다 차지할 정도로 아주 크게 제목을 적어줘. 오른쪽에는 초코우유 이미지 크게 보여줘. 그리고 그림 아래 설명을 간략히 적어줘."
                )
                
                image_input = gr.File(
                    label="이미지 파일들",
                    file_count="multiple",
                    file_types=["image"]
                )
                
                generate_btn = gr.Button("레이아웃 생성", variant="primary")
            
            with gr.Column():
                # output_text = gr.Textbox(
                #     label="결과",
                #     lines=4,
                #     interactive=False
                # )
                output_image = gr.Image(
                    label="생성된 레이아웃 이미지",
                    type="pil"
                )
        
        generate_btn.click(
            fn=run_layout_generation,
            inputs=[query_input, image_input],
            outputs=[output_image]
        )
        
        # 예시 섹션
        gr.Markdown("## 📝 사용 예시")
        gr.Markdown("""
        ### 쿼리 예시:
        - "빙그레 초코우유에 대한 카드뉴스를 제작할거야. 제목은 '초코 타임!'이야, 정중앙에 크게 제목이 있고 두장이 살짝만 겹쳐서 제목 아래에 사진이 위치하고 제목 위에는 설명을 간단히 써줘."
        - "빙그레 초코 우유에 대한 홍보 카드뉴스를 만들거야. 왼쪽 면을 거의 다 차지할 정도로 아주 크게 제목을 적어줘."
        - "제목은 '초코 타임!'이야, 정중앙에 크게 제목이 있고 두장이 살짝만 겹쳐서 제목 아래에 사진이 위치해줘."
        - "상단에 로고, 중앙에 큰 제목, 하단에 이미지 2장을 나란히 배치해줘."
        """)
    
    return demo

if __name__ == "__main__":
    # Gradio 인터페이스 실행
    demo = create_gradio_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)