from bs4 import BeautifulSoup as Soup
from pathlib import Path
import yaml

def extract_html_only(mixed_text: str) -> str:
    """
    mixed_text 안에서 최상위 태그 노드만 골라서 합쳐서 반환.
    순수 HTML 블록만 걸러내기 위해 recursive=False 사용.
    """
    soup = Soup(mixed_text, "html.parser")
    html_segments = [
        str(node)
        for node in soup.contents
        if getattr(node, "name", None)
    ]
    return "".join(html_segments)


def generate_and_save_html(
    html_content: str,
    output_dir: str = "generated_html",
    open_browser: bool = True,
):
    # 1) HTML만 추출
    clean_html = extract_html_only(html_content)

    # 2) 파싱 & prettify
    soup = Soup(clean_html, "html.parser")
    formatted_html = soup.prettify()

    # 3) 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 4) 파일 저장
    file_path = output_path / "output.html"
    file_path.write_text(formatted_html, encoding="utf-8")
    print(f"✅ HTML 파일 생성: {file_path}")

    # 5) 브라우저 열기
    if open_browser:
        import webbrowser

        webbrowser.open(f"file://{file_path.absolute()}")

    return file_path


def load_system_template(yaml_path: str) -> str:
    config_path = Path(yaml_path)
    prompt_data = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    return prompt_data['template']
