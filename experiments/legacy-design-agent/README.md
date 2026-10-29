# Legacy Design Agent

자연어 요청에서 디자인 컨셉을 기획하고 HTML 홍보물로 표현하던 초기 실험을 보존한 폴더입니다. 현재 카드뉴스 생성 파이프라인과 독립된 참고용 스냅샷입니다.

## 출처

- 원본 저장소: `Yunseo-Lab/proj-say-it-see-it`
- 원본 커밋: `55be50195365bcdef5e510ffeee6c224d3d3ab26` (2025-05-25 UTC)
- 보존일: 2026-10-29
- 라이선스: 원본의 [MIT License](LICENSE) 유지
- [SOURCE.json](SOURCE.json)에 보존 파일별 원본 Git blob SHA와 SHA-256을 기록했습니다. 보존한 원본 파일의 내용은 수정하지 않았습니다.

## 보존한 실험

`runner.py` → `graph_builder.py` → `designer` → `layout` → `background` → `drawer` 순서로 진행하고, 마지막 응답을 HTML 파일로 저장합니다.

- `nodes/designer.py`, `prompts/designer_prompt.yaml`: 디자인 컨셉, 색상, 폰트, 배경 방향, 레이아웃 방향, 장식 아이콘을 `DesignSpec`으로 구조화하는 실험
- `nodes/layout.py`, `prompts/layout_prompt.yaml`: 캔버스 크기와 디자인 방향을 받아 요소별 `[x, y, width, height]` 좌표를 생성하는 실험
- `nodes/background.py`: 디자인에서 선택한 첫 번째 색상을 배경으로 사용하는 임시 구현
- `nodes/drawer.py`, `prompts/drawer_prompt.yaml`, `utils.py`: 디자인 스펙과 좌표를 HTML/CSS로 표현하고 `generated_html/output.html`로 저장하는 실험
- `graph_builder.py`, `nodes/common.py`: LangGraph 상태, 도구, 메모리 및 노드 연결
- `tools/`: 원본 `nodes/common.py`의 import를 보존하기 위해 함께 포함한 구형 도구. `background.py`, `layout.py`는 원본에서 deprecated로 표시되어 있습니다.
- `pyproject.toml`, `uv.lock`, `.python-version`: 당시 의존성 및 Python 버전 기록

## 재현 시 참고

Python 3.12와 `uv`가 있는 별도 환경에서 이 폴더를 작업 디렉터리로 사용합니다.

```bash
cd experiments/legacy-design-agent
uv sync --frozen
# 로컬 .env에 OPENAI_API_KEY와 TAVILY_API_KEY 설정
uv run --frozen python runner.py
```

`runner.py`의 예제 요청과 캔버스 크기를 수정해 입력을 바꿀 수 있습니다. 실행 시 외부 OpenAI API와 Tavily 설정을 사용하며, HTML을 저장한 뒤 기본 브라우저를 엽니다. API 키와 생성 결과는 커밋하지 않습니다.

이관 시 검증한 것은 Python 구문, 원본 파일 일치 여부와 보존 파일 내 인증정보 패턴입니다. 의존성 설치 및 외부 API를 사용하는 전체 실행은 검증하지 않았습니다. 위 명령은 원본 실행 방법을 설명하며 정상 실행을 보장하지 않습니다.

## 알려진 한계

- `designer`에서 도구 바인딩과 구조화 출력을 연결하는 방식은 설치된 LangChain 버전과 호환성을 확인해야 합니다.
- `designer_prompt.yaml`은 Markdown 기획안을 요청하지만 노드는 구조화 출력을 요청합니다. 재사용할 때 출력 규약을 정리해야 합니다.
- `drawer`의 도구 바인딩은 주석 처리되어 있습니다. 그래프에 도구 노드가 있어도 완성된 검색·도구 실행 기능으로 간주할 수 없습니다.
- 배경은 실제 이미지 생성이 아닌 임시 색상 처리입니다.
- `tools/layout.py`에는 원본 개발 환경의 절대 프롬프트 경로가 남아 있습니다. 현재 실험의 레이아웃 노드는 `nodes/layout.py`이며, 구형 도구를 직접 재사용하려면 경로를 수정해야 합니다.
- 전체 흐름은 초기 프로토타입이며, 현재 `LayoutGenerator/` API나 Flutter 웹 UI에 연결하지 않았습니다.

## 제외한 자료

중복 모듈(`nodes.py`, `nodes/_layout.py`), `lagacy/MVP*.py` 설계 초안, 튜토리얼·중복 노트북, 기존 LayoutPrompter 구현과 WebUI 데이터, 생성 결과물, `.gradio` 상태, 인증서 및 Python 캐시는 포함하지 않았습니다. 현재 카드뉴스용 구현은 저장소의 `LayoutGenerator/`를 사용합니다.

원본 저장소의 전체 Git 이력은 삭제 전 별도의 비공개 로컬 Git bundle로 백업했습니다. 이 폴더는 그 전체 백업을 대체하지 않습니다.
