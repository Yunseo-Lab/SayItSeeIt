본 LayoutGenerator는 Microsoft의 LayoutPrompter를 기반으로 제작되었다.
LayoutPrompter는 다양한 제약조건 기반의 레이아웃 생성 작업들을 통합적으로 해결하기 위해 고안된 프롬프트 기반의 프레임워크로, 학습 없이도 고품질의 그래픽 레이아웃을 생성할 수 있는 데이터 효율적 방법이다. 본 프로젝트에서는 LayoutPrompter의 핵심 구성 요소인 입출력 직렬화(Input-Output Serialization), 동적 예시 선택(Dynamic Exemplar Selection), **레이아웃 랭커(Layout Ranker)**를 그대로 수용하되, 사용자의 자연어 요구를 보다 직관적으로 반영할 수 있도록 Text-to-Layout → Refinement 조합을 중심으로 설계되었다.

이 조합은 먼저 자연어 설명을 통해 레이아웃 구성 요소를 정의하고 초기 배치를 생성한 후, 미세 조정 단계를 통해 시각적 정렬과 겹침 최소화 등의 요소를 보완한다. 특히 Text-to-Layout 단계에서는 HTML 기반의 직렬화 포맷을 채택하여, 사전학습된 LLM이 이미 학습한 HTML 문법 지식을 활용할 수 있게 하였으며, Refinement 단계에서는 정렬 정확도(Alignment), 요소 간 겹침(Overlap), mIoU 등의 평가 지표를 종합하여 레이아웃 품질을 향상시킨다.

본 시스템은 레이아웃 설계에 대한 전문 지식이 없는 사용자도 간단한 자연어 입력만으로 현대적이고 미적으로 균형 잡힌 레이아웃을 설계할 수 있도록 지원하며, 모바일 UI, 포스터, 웹페이지 등 다양한 도메인에 적용 가능하다. 또한 LLM 기반의 zero-shot/few-shot 학습 특성을 활용해 새로운 작업 유형에도 빠르게 확장될 수 있는 유연성을 갖추고 있다.

**layout generation task**: `Gen-T`, `Gen-TS`, `Gen-R`, `Completion`, `Refinement`, `Content-Aware`, `Text-to-Layout` 

## 🧩 LayoutPrompter의 주요 Task 분류 및 정의

| Task 이름            | 정식 명칭                                         | 의미 및 설명                                                                                                   |
| ------------------ | --------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Gen-T**          | **Generation Conditioned on Types**           | 요소들의 \*\*종류(type)\*\*만 주어졌을 때, 이를 바탕으로 레이아웃을 생성하는 작업<br>예: "image, text, button"만 주어지고, 위치/크기 없이 배치       |
| **Gen-TS**         | **Generation Conditioned on Types and Sizes** | 요소들의 \*\*종류(type)\*\*와 \*\*크기(size)\*\*가 주어졌을 때, 이를 고려해 요소를 적절히 배치하는 작업<br>예: "text 90x12, image 60x40" 등 |
| **Gen-R**          | **Generation Conditioned on Relationships**   | 요소 간의 \*\*관계(위치적 제약)\*\*가 주어졌을 때, 이를 만족하는 레이아웃 생성<br>예: "A는 B 위에 있어야 한다", "C는 D보다 작아야 한다" 등               |
| **Completion**     | **Layout Completion**                         | **일부 요소만 주어진 상태**에서 전체 레이아웃을 완성하는 작업<br>→ 주어진 요소를 유지하면서 나머지를 자연스럽게 배치                                     |
| **Refinement**     | **Layout Refinement**                         | 기존 레이아웃을 **미세 조정**하여 정렬, 겹침 최소화, 시각적 품질 향상 등 개선<br>→ 초기 레이아웃이 존재함                                         |
| **Content-Aware**  | **Content-Aware Layout Generation**           | 입력 이미지(배경 등)의 **내용을 고려하여** 요소를 배치하는 작업<br>→ 예: 사람이 있는 영역을 피해서 텍스트 배치                                      |
| **Text-to-Layout** | **Natural Language to Layout**                | 자연어로 된 설명을 받아서 거기에 맞는 레이아웃을 생성하는 작업<br>→ 예: “로고와 설명이 있는 페이지” 요청을 HTML 형태로 출력                              |

---

### 예시1 (Gen-T):

```
Input: [text, image, button]
→ Output: text는 좌상단, image는 가운데, button은 우하단
```

### 예시2 (Gen-R):

```
Input: [text1 위에 image, image보다 큰 button]
→ Output: 이러한 관계를 만족하는 좌표로 배치
```

### 예시3 (Text-to-Layout):

```
Input: "이 페이지에는 큰 로고, 제목, 버튼이 있어야 합니다."
→ Output: HTML 형태로 로고, 제목, 버튼을 실제 위치/크기 포함하여 배치
```

## 레이아웃 생성 Task 조합

**Text-to-Layout → Refinement**

### ✅ 조합 선정 이유

이 조합은 **자연어 기반의 직관적인 레이아웃 구성**과, **정렬 및 시각적 완성도를 높이는 후처리 단계**를 결합함으로써, **비전문가도 고품질의 현대적인 UI/UX 레이아웃을 빠르게 생성**할 수 있도록 한다.

1. **Text-to-Layout**은 사용자의 자연어 설명만으로 전체 레이아웃의 구조를 자동으로 생성한다.
   예를 들어 “로고, 제목, 설명, 버튼이 포함된 페이지”와 같은 입력에 대해, LLM은 해당 구성 요소들을 적절한 HTML 형태로 배치하여 초기 초안을 만들어낸다. 이 과정은 사용자의 의도를 고수준에서 빠르게 반영할 수 있어 **빠른 프로토타이핑에 유리**하다.

2. 하지만 Text-to-Layout 단계에서 생성된 레이아웃은 **정렬이 미묘하게 어긋나거나 시각적으로 불균형할 수 있다.** 이를 해결하기 위해 **Refinement** 단계를 추가한다.
   Refinement는 LLM이 만든 초안을 기반으로 요소 간 **정렬(Alignment), 겹침 최소화(Overlap), 여백 조정** 등의 미세 조정을 수행하여, **시각적 완성도를 극대화**한다.

3. 이 조합은 기존 모델들과 달리 사전 학습이나 파인튜닝 없이도 작동하며, LLM의 **in-context learning 능력만으로도 다양한 요구를 만족하는 고품질 레이아웃을 생성**할 수 있다는 점에서 매우 실용적이다.

### ✅ 활용 예시

* 웹페이지 초기 설계
* 모바일 앱의 온보딩 화면
* 마케팅 배너 및 카드형 UI 생성
* 비전문가도 활용 가능한 레이아웃 자동화 툴

---

이 조합은 특히 \*\*직관성(사용자 의도 반영)\*\*과 \*\*완성도(디자인 정제)\*\*의 균형을 잘 맞춘다는 점에서, **현대적이고 실용적인 레이아웃 생성 전략**으로 평가된다.

## Citation

[LayoutPrompter](https://arxiv.org/pdf/2311.06495.pdf) is a versatile method for graphic layout generation, capable of solving various conditional layout generation tasks (as illustrated on the left side) across a range of layout domains (as illustrated on the right side) without any model training or fine-tuning.

```
@inproceedings{lin2023layoutprompter,
  title={LayoutPrompter: Awaken the Design Ability of Large Language Models},
  author={Lin, Jiawei and Guo, Jiaqi and Sun, Shizhao and Yang, Zijiang James and Lou, Jian-Guang and Zhang, Dongmei},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
