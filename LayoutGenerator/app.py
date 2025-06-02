"""
카드뉴스 생성기 - Gradio 웹 인터페이스
텍스트 요청사항과 이미지를 통해 카드뉴스 레이아웃을 생성합니다.
"""

import os
import time
import hashlib
import shutil
import threading
from io import BytesIO
from PIL import Image

import gradio as gr
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import invoke_graph, random_uuid

from main import TextToLayoutPipeline, GraphState, initialize_graph
from src.generators.copy_generator import generate_copies

# 설정 상수들
DATASET_NAME = "cardnews"
IMAGES_DIR = "src/images"
OUTPUT_IMAGE_PATH = "output/output_poster.png"
MAX_IMAGE_CHANGES = 2
MAX_WAIT_TIME = 20
POLLING_INTERVAL = 0.1
RECURSION_LIMIT = 20
SERVER_PORT = 7860

# 파이프라인 초기화
pipeline = TextToLayoutPipeline(dataset=DATASET_NAME)


def copy_image_files(image_files):
    """업로드된 이미지 파일들을 로컬 디렉토리로 복사"""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    image_filenames = []
    
    if image_files:
        for file in image_files:
            filename = os.path.basename(file.name)
            dest_path = os.path.join(IMAGES_DIR, filename)
            
            if not os.path.exists(dest_path):
                try:
                    shutil.copy2(file.name, dest_path)
                    print(f"이미지 복사됨: {filename} -> {dest_path}")
                except Exception as e:
                    print(f"이미지 복사 실패: {filename}, 오류: {e}")
            
            image_filenames.append(filename)
    
    return image_filenames


def run_graph_in_background(initial_state, config):
    """백그라운드에서 그래프 실행"""
    def run_graph():
        try:
            app = initialize_graph()
            invoke_graph(app, initial_state, config)
        except Exception:
            pass
    
    thread = threading.Thread(target=run_graph)
    thread.start()
    return thread


def get_image_hash(image_path):
    """이미지 파일의 MD5 해시값을 계산"""
    try:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def load_image_from_path(image_path):
    """이미지 파일을 PIL Image 객체로 로드"""
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        return Image.open(BytesIO(img_bytes)).copy()
    except Exception:
        return None

def run_layout_generation(query_text, image_files):
    """
    Gradio에서 호출할 메인 함수 - 레이아웃 생성 및 실시간 이미지 업데이트
    
    Args:
        query_text (str): 사용자의 레이아웃 요청사항
        image_files (list): 업로드된 이미지 파일들
        
    Yields:
        PIL.Image: 생성된 레이아웃 이미지들
    """
    # 이미지 파일 복사 및 준비
    image_filenames = copy_image_files(image_files)
    
    # 초기 상태 설정
    initial_state: GraphState = {
        "pipeline": pipeline,
        "query": query_text,
        "layout": [],
        "copy": [],
        "images": image_filenames,
    }
    
    # 그래프 실행 설정
    config = RunnableConfig(
        recursion_limit=RECURSION_LIMIT,
        configurable={"thread_id": random_uuid()}
    )
    
    # 백그라운드에서 그래프 실행
    run_graph_in_background(initial_state, config)
    
    # 이미지 변경 감지 및 실시간 업데이트
    yield from monitor_image_changes()


def monitor_image_changes():
    """출력 이미지의 변경사항을 감지하고 업데이트된 이미지를 반환"""
    yielded_hashes = set()
    changed_count = 0
    waited = 0
    prev_img_hash = None
    first_hash_set = False
    
    while changed_count < MAX_IMAGE_CHANGES and waited < MAX_WAIT_TIME:
        current_hash = get_image_hash(OUTPUT_IMAGE_PATH)
        
        if current_hash:
            # 첫 번째 해시는 캐시로 간주하고 건너뛰기
            if not first_hash_set:
                prev_img_hash = current_hash
                first_hash_set = True
            # 새로운 이미지가 생성되었을 때만 반환
            elif current_hash not in yielded_hashes and current_hash != prev_img_hash:
                img = load_image_from_path(OUTPUT_IMAGE_PATH)
                if img:
                    yield img
                    yielded_hashes.add(current_hash)
                    changed_count += 1
        else:
            # 이미지가 없고 첫 번째 변경이라면 None 반환
            if changed_count == 0:
                yield None
        
        time.sleep(POLLING_INTERVAL)
        waited += POLLING_INTERVAL
    
    # 마지막 이미지 한 번 더 반환 (Gradio 안정성을 위해)
    if changed_count == 1:
        final_img = load_image_from_path(OUTPUT_IMAGE_PATH)
        if final_img:
            yield final_img

def create_input_components():
    """입력 컴포넌트들을 생성"""
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
    
    return query_input, image_input, generate_btn


def create_example_section():
    """예시 섹션을 생성"""
    gr.Markdown("## 📝 사용 예시")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 쿼리 예시:
            - "빙그레 초코우유에 대한 카드뉴스를 제작할거야. 제목은 '초코 타임!'이야, 정중앙에 크게 제목이 있고 두장이 살짝만 겹쳐서 제목 아래에 사진이 위치하고 제목 위에는 설명을 간단히 써줘."
            - "빙그레 초코 우유에 대한 홍보 카드뉴스를 만들거야. 왼쪽 면을 거의 다 차지할 정도로 아주 크게 제목을 적어줘."
            - "제목은 '초코 타임!'이야, 정중앙에 크게 제목이 있고 두장이 살짝만 겹쳐서 제목 아래에 사진이 위치해줘."
            - "상단에 로고, 중앙에 큰 제목, 하단에 이미지 2장을 나란히 배치해줘."
            """)
        
        with gr.Column():
            gr.Markdown("#### 예시 이미지:")
            gr.Image(
                value="src/images/choco1.png",
                label="예시 이미지",
                show_label=True,
                height=300
            )


def create_gradio_interface():
    """Gradio 웹 인터페이스를 생성"""
    with gr.Blocks(title="카드뉴스 생성기") as demo:
        # 헤더
        gr.Markdown("# 🎨 카드뉴스 생성기")
        gr.Markdown("원하는 카드뉴스 레이아웃을 설명하고 이미지를 업로드하세요!")
        
        # 메인 인터페이스
        with gr.Row():
            # 입력 섹션
            with gr.Column():
                query_input, image_input, generate_btn = create_input_components()
            
            # 출력 섹션
            with gr.Column():
                output_image = gr.Image(
                    label="생성된 레이아웃 이미지",
                    type="pil"
                )
        
        # 이벤트 연결
        generate_btn.click(
            fn=run_layout_generation,
            inputs=[query_input, image_input],
            outputs=[output_image]
        )
        
        # 예시 섹션
        create_example_section()
    
    return demo


def main():
    """메인 실행 함수"""
    demo = create_gradio_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=SERVER_PORT
    )

if __name__ == "__main__":
    main()