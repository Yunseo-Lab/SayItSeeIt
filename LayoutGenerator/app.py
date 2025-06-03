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
MAX_WAIT_TIME = 60
POLLING_INTERVAL = 0.2
RECURSION_LIMIT = 20
SERVER_PORT = 7860

# 파이프라인 초기화
pipeline = TextToLayoutPipeline(dataset=DATASET_NAME)


def convert_images_to_bytes(image_files):
    """업로드된 이미지 파일들을 바이트 데이터로 변환"""
    image_data_list = []
    
    if image_files:
        for file in image_files:
            try:
                with open(file.name, "rb") as f:
                    image_data = f.read()
                image_data_list.append(image_data)
                print(f"이미지 바이트 데이터 로드됨: {os.path.basename(file.name)}")
            except Exception as e:
                print(f"이미지 로드 실패: {os.path.basename(file.name)}, 오류: {e}")
    
    return image_data_list


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

def run_layout_generation(query_text, image_files, logo_file):
    """
    Gradio에서 호출할 메인 함수 - 레이아웃 생성 및 실시간 이미지 업데이트
    
    Args:
        query_text (str): 사용자의 레이아웃 요청사항
        image_files (list): 업로드된 이미지 파일들
        logo_file: 업로드된 로고 파일
        
    Yields:
        PIL.Image: 생성된 레이아웃 이미지들
    """
    # 이전 출력 이미지 파일 삭제 (이전 레이아웃이 보이는 것을 방지)
    if os.path.exists(OUTPUT_IMAGE_PATH):
        try:
            os.remove(OUTPUT_IMAGE_PATH)
            print(f"이전 출력 이미지 삭제됨: {OUTPUT_IMAGE_PATH}")
        except Exception as e:
            print(f"이전 출력 이미지 삭제 실패: {e}")
    
    # 이미지 파일을 바이트 데이터로 변환
    image_data_list = convert_images_to_bytes(image_files)
    
    # 로고 데이터 처리
    logo_data = None
    if logo_file:
        try:
            with open(logo_file.name, "rb") as f:
                logo_data = f.read()
            print(f"로고 파일 로드됨: {logo_file.name}")
        except Exception as e:
            print(f"로고 파일 로드 실패: {e}")
    
    # 초기 상태 설정
    initial_state: GraphState = {
        "pipeline": pipeline,
        "query": query_text,
        "layout": [],
        "copy": [],
        "images": image_data_list,
        "logo_data": logo_data,
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


def run_layout_generation_with_status(query_text, image_files, logo_file):
    """
    상태 메시지와 함께 레이아웃 생성을 실행하는 래퍼 함수
    """
    # 첫 번째로 로딩 상태 표시
    yield "🔄 레이아웃 생성을 시작합니다...", None
    
    # 실제 레이아웃 생성 실행
    image_count = 0
    for img in run_layout_generation(query_text, image_files, logo_file):
        if img is None:
            if image_count == 0:
                yield "⏳ 레이아웃을 생성 중입니다... 잠시만 기다려주세요.", None
        else:
            image_count += 1
            if image_count == 1:
                yield "✨ 레이아웃이 생성되었습니다! 이제 문구와 이미지를 추가합니다~!", img
            else:
                yield "🎨 문구와 이미지를 추가하였습니다!", img
    
    # 최종 완료 상태
    if image_count > 0:
        yield "✅ 레이아웃 생성이 완료되었습니다!", img
    else:
        yield "❌ 레이아웃 생성에 실패했습니다. 다시 시도해주세요.", None


def monitor_image_changes():
    """출력 이미지의 변경사항을 감지하고 업데이트된 이미지를 반환"""
    yielded_hashes = set()
    changed_count = 0
    waited = 0
    loading_shown = False
    
    # 시작 시점에 이미지가 없다는 것을 확인
    initial_hash = get_image_hash(OUTPUT_IMAGE_PATH)
    if initial_hash:
        print(f"경고: 시작 시점에 이미지가 이미 존재함 (해시: {initial_hash[:8]}...)")
    
    # 첫 번째로 로딩 상태를 표시 (None 반환)
    print("레이아웃 생성을 시작합니다...")
    yield None
    loading_shown = True
    
    while changed_count < MAX_IMAGE_CHANGES and waited < MAX_WAIT_TIME:
        current_hash = get_image_hash(OUTPUT_IMAGE_PATH)
        
        if current_hash and current_hash not in yielded_hashes:
            # 새로운 이미지가 생성되었음
            img = load_image_from_path(OUTPUT_IMAGE_PATH)
            if img:
                print(f"새 이미지 감지됨 (해시: {current_hash[:8]}...)")
                yield img
                yielded_hashes.add(current_hash)
                changed_count += 1
        
        time.sleep(POLLING_INTERVAL)
        waited += POLLING_INTERVAL
    
    # 타임아웃 또는 최대 변경 횟수 도달 시 마지막 이미지 확인
    if changed_count > 0:
        final_img = load_image_from_path(OUTPUT_IMAGE_PATH)
        if final_img:
            print(f"최종 이미지 반환 (총 {changed_count}회 변경)")
            yield final_img
    else:
        print(f"이미지 생성 타임아웃 ({MAX_WAIT_TIME}초 대기 완료)")
        # 타임아웃 시에도 None을 반환하여 사용자에게 상태를 알림
        yield None

def create_input_components():
    """입력 컴포넌트들을 생성"""
    query_input = gr.Textbox(
        label="레이아웃 요청사항",
        placeholder="예: 빙그레 초코우유에 대한 카드뉴스를 제작할거야. 제목은 '초코 타임!'이야...",
        lines=4,
        value="빙그레 초코 우유에 대한 홍보 카드뉴스를 만들거야. 왼쪽 면을 거의 다 차지할 정도로 아주 크게 제목을 적어줘. 오른쪽에는 초코우유 이미지 크게 보여줘. 그리고 그림 아래 설명을 간략히 적어줘. 죄측 하단에는 로고를 넣어줘."
    )
    
    image_input = gr.File(
        label="이미지 파일들",
        file_count="multiple",
        file_types=["image"]
    )
    
    logo_input = gr.File(
        label="로고 파일 (PNG)",
        file_count="single",
        file_types=[".png"]
    )
    
    generate_btn = gr.Button("레이아웃 생성", variant="primary")
    
    return query_input, image_input, logo_input, generate_btn


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
            
            ### 파일 업로드:
            - **이미지**: 카드뉴스에 사용할 이미지들을 업로드하세요
            - **로고**: PNG 형식의 로고 파일을 업로드하세요 (선택사항)
            """)
        
        with gr.Column():
            gr.Markdown("#### 예시 제품 이미지:")
            gr.Image(
                value="src/images/chocomilk1.png",
                label="빙그레 초코타임",
                show_label=True,
                height=150
            )
            gr.Markdown("#### 예시 로고 이미지:")
            gr.Image(
                value="src/images/logo.png",
                label="빙그레 로고",
                show_label=True,
                height=150
            )


def create_gradio_interface():
    """Gradio 웹 인터페이스를 생성"""
    with gr.Blocks(title="카드뉴스 생성기") as demo:
        # 헤더
        gr.Markdown("# 🎨 카드뉴스 생성기")
        gr.Markdown("원하는 카드뉴스 레이아웃을 설명하고 이미지와 로고를 업로드하세요!")
        
        # 메인 인터페이스
        with gr.Row():
            # 입력 섹션
            with gr.Column():
                query_input, image_input, logo_input, generate_btn = create_input_components()
            
            # 출력 섹션
            with gr.Column():
                output_image = gr.Image(
                    label="생성된 레이아웃 이미지",
                    type="pil",
                    placeholder="레이아웃 생성 버튼을 클릭하여 시작하세요"
                )
                
                # 상태 메시지 표시
                status_text = gr.Textbox(
                    label="생성 상태",
                    value="대기 중...",
                    interactive=False,
                    visible=True
                )
        
        # 이벤트 연결
        generate_btn.click(
            fn=run_layout_generation_with_status,
            inputs=[query_input, image_input, logo_input],
            outputs=[status_text, output_image]
        )
        
        # 예시 섹션
        create_example_section()
    
    return demo


def main():
    """메인 실행 함수"""
    demo = create_gradio_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=SERVER_PORT
    )

if __name__ == "__main__":
    main()