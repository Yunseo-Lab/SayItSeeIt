# SayItSeeIt

Generate promotional visuals from natural-language requests and images, built on LayoutPrompter.

자연어 요청과 이미지를 입력받아 홍보용 카드뉴스 이미지를 생성하는 프로젝트입니다. Flutter 웹 UI와 FastAPI 서버를 연결하고, 레이아웃 생성·문구 생성·이미지 렌더링을 하나의 파이프라인으로 구성했습니다.

## How it works

1. 사용자가 자연어 요청과 이미지, 선택적으로 로고를 전달합니다.
2. LayoutPrompter 기반 파이프라인이 레이아웃을 생성합니다.
3. 레이아웃과 요청에 맞는 문구를 생성합니다.
4. 이미지와 문구를 배치해 결과 이미지를 반환합니다.

## Structure

- `LayoutGenerator/` — Python 생성 파이프라인과 FastAPI 서버
- `say_it_see_it_web/` — Flutter 웹 인터페이스
- `POST /api/cardnews` — 요청과 이미지를 받아 생성 결과를 반환하는 API

## Run locally

아래 명령은 Flutter와 Python 프로젝트 의존성 및 모델/API 설정이 준비된 환경을 기준으로 합니다.

### Web interface

```bash
cd say_it_see_it_web
flutter build web
cd build/web
python3 -m http.server 8080 --bind 0.0.0.0
```

### API server

별도 터미널에서 실행합니다.

```bash
cd LayoutGenerator
python api_server.py
```

## Technical background

레이아웃 생성은 Microsoft의 [LayoutPrompter](https://github.com/microsoft/LayoutGeneration/tree/main/LayoutPrompter)를 기반으로 합니다. 기존 연구를 활용한 부분과 Text-to-Layout / Refinement 구성은 [생성 파이프라인 문서](LayoutGenerator/README.md)에 설명되어 있습니다.

이 저장소는 프로토타입입니다. 위 실행 명령은 기존 저장소에 기록된 절차이며, 이 문서 정리 과정에서 전체 생성 파이프라인의 실행을 재검증하지는 않았습니다.
