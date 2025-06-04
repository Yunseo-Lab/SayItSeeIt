import 'package:flutter/material.dart';

class ExampleSection extends StatelessWidget {
  const ExampleSection({super.key});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.grey[100],
      margin: const EdgeInsets.symmetric(vertical: 10),
      child: const Padding(
        padding: EdgeInsets.all(14.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              "📝 사용 예시",
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
            ),
            SizedBox(height: 6),
            Text("""
- "빙그레 초코우유에 대한 카드뉴스를 제작할거야. 제목은 '초코 타임!'이야, 정중앙에 크게 제목이 있고 두장이 살짝만 겹쳐서 제목 아래에 사진이 위치하고 제목 위에는 설명을 간단히 써줘."
- "빙그레 초코 우유에 대한 홍보 카드뉴스를 만들거야. 왼쪽 면을 거의 다 차지할 정도로 아주 크게 제목을 적어줘."
- "제목은 '초코 타임!'이야, 정중앙에 크게 제목이 있고 두장이 살짝만 겹쳐서 제목 아래에 사진이 위치해줘."
- "상단에 로고, 중앙에 큰 제목, 하단에 이미지 2장을 나란히 배치해줘."
"""),
            SizedBox(height: 6),
            Text("예시 이미지와 로고(화면 구성에 추가 가능)"),
            // 실제 예시 이미지는 asset/네트워크 이미지로 확장 가능
          ],
        ),
      ),
    );
  }
}
