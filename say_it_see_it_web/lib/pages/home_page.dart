import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_saver/file_saver.dart';
import '../services/api_service.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _queryController = TextEditingController(
    text:
        '빙그레 초코 우유에 대한 홍보 카드뉴스를 만들거야. 왼쪽 면을 거의 다 차지할 정도로 아주 크게 제목을 적어줘. 오른쪽에는 초코우유 이미지 크게 보여줘. 그리고 그림 아래 설명을 간략히 적어줘.',
  );
  List<XFile> _images = [];
  XFile? _logo;
  String _status = '대기 중...';
  Uint8List? _resultImage;
  bool _isLoading = false;

  Future<void> _pickImages() async {
    final images = await ImagePicker().pickMultiImage();
    if (images != null) setState(() => _images = images);
  }

  Future<void> _pickLogo() async {
    final logo = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (logo != null) setState(() => _logo = logo);
  }

  Future<void> _generateLayout() async {
    setState(() {
      _isLoading = true;
      _status = '🔄 레이아웃 생성 중...';
    });

    final result = await ApiService.generateCardNews(
      query: _queryController.text,
      imageFiles: _images,
      logoFile: _logo,
    );

    setState(() {
      _isLoading = false;
      if (result.success) {
        _resultImage = result.imageBytes;
        _status = result.statusMessage ?? '생성 성공!';
      } else {
        _status = result.statusMessage ?? '❌ 레이아웃 생성 실패';
      }
    });
  }

  Widget _exampleCard() {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 10, horizontal: 6),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 18,
            spreadRadius: 2,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: const [
              Icon(
                Icons.tips_and_updates_rounded,
                color: Colors.orange,
                size: 26,
              ),
              SizedBox(width: 7),
              Text(
                '사용 예시',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
              ),
            ],
          ),
          const SizedBox(height: 11),
          const Text(
            '쿼리 예시:',
            style: TextStyle(fontWeight: FontWeight.w600, fontSize: 15),
          ),
          const SizedBox(height: 2),
          const Text(
            "- 빙그레 초코 우유에 대한 홍보 카드뉴스를 만들거야. 왼쪽 면을 거의 다 차지할 정도로 아주 크게 제목을 적어줘. 오른쪽에는 초코우유 이미지 크게 보여줘. 그리고 그림 아래 설명을 간략히 적어줘.\n"
            "- 빙그레 초코우유에 대한 카드뉴스를 제작할거야. 제목은 '초코 타임!'이야, 정중앙에 크게 제목이 있고 두장이 살짝만 겹쳐서 제목 아래에 사진이 위치하고 제목 위에는 설명을 간단히 써줘.\n"
            "- 제목은 '초코 타임!'이야, 정중앙에 크게 제목이 있고 두장이 살짝만 겹쳐서 아래에 사진이 위치하게 해줘.\n"
            "- 상단엔 로고, 중앙엔 제목, 하단에 이미지 2장을 나란히 배치해줘.",
            style: TextStyle(fontSize: 13, height: 1.6),
          ),
          const SizedBox(height: 9),
          const Text(
            '파일 업로드:',
            style: TextStyle(fontWeight: FontWeight.w600, fontSize: 15),
          ),
          const Text(
            '이미지: 카드뉴스에 사용할 이미지들을 업로드하세요\n로고: PNG 형식의 로고 파일을 업로드하세요 (선택사항)',
            style: TextStyle(fontSize: 13, height: 1.5),
          ),
        ],
      ),
    );
  }

  Widget _inputCard() {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 10, horizontal: 6),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.07),
            blurRadius: 18,
            spreadRadius: 2,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 쿼리 입력
          TextField(
            controller: _queryController,
            maxLines: 3,
            style: const TextStyle(fontSize: 15),
            decoration: const InputDecoration(
              labelText: "프롬프트",
              border: OutlineInputBorder(),
              isDense: true,
              contentPadding: EdgeInsets.all(12),
            ),
          ),
          const SizedBox(height: 15),
          // 이미지 업로드 + 미리보기
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  icon: const Icon(Icons.image, color: Colors.blue),
                  label: const Text("이미지 업로드"),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.blue,
                    side: const BorderSide(color: Colors.blue),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(13),
                    ),
                  ),
                  onPressed: _pickImages,
                ),
              ),
              const SizedBox(width: 12),
              if (_images.isNotEmpty)
                SizedBox(
                  height: 36,
                  child: Row(
                    children: _images
                        .map(
                          (img) => Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 2),
                            child: Image.network(img.path, height: 34),
                          ),
                        )
                        .toList(),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 8),
          // 로고 업로드 + 미리보기
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  icon: const Icon(Icons.upload_rounded, color: Colors.teal),
                  label: const Text("로고 업로드"),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.teal,
                    side: const BorderSide(color: Colors.teal),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(13),
                    ),
                  ),
                  onPressed: _pickLogo,
                ),
              ),
              const SizedBox(width: 12),
              if (_logo != null) Image.network(_logo!.path, height: 34),
            ],
          ),
          const SizedBox(height: 15),
          // 생성 버튼
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: _isLoading ? null : _generateLayout,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.deepOrange,
                foregroundColor: Colors.white,
                textStyle: const TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 17,
                ),
                padding: const EdgeInsets.symmetric(vertical: 15),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(13),
                ),
                elevation: 2,
              ),
              child: _isLoading
                  ? const CircularProgressIndicator(color: Colors.white)
                  : const Text('레이아웃 생성'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _outputCard() {
    if (_resultImage == null) return const SizedBox.shrink();
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 12, horizontal: 6),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.08),
            blurRadius: 12,
            spreadRadius: 2,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          GestureDetector(
            onTap: () {
              showDialog(
                context: context,
                builder: (_) => Dialog(
                  child: InteractiveViewer(
                    child: Image.memory(_resultImage!),
                  ),
                ),
              );
            },
            child: Image.memory(_resultImage!, height: 300),
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              IconButton(
                icon: const Icon(Icons.zoom_in),
                tooltip: "이미지 확대",
                onPressed: () {
                  showDialog(
                    context: context,
                    builder: (_) => Dialog(
                      child: InteractiveViewer(
                        child: Image.memory(_resultImage!),
                      ),
                    ),
                  );
                },
              ),
              IconButton(
                icon: const Icon(Icons.download),
                tooltip: "이미지 저장",
                onPressed: () async {
                  await FileSaver.instance.saveFile(
                    name: 'output_image',
                    bytes: _resultImage!,
                    ext: 'png',
                    mimeType: MimeType.png,
                  );
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text("저장(다운로드) 완료!")),
                  );
                },
              ),
            ],
          ),
          Text(_status, style: const TextStyle(fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }

  Widget _mainTitle() {
    return Container(
      margin: const EdgeInsets.only(top: 36, bottom: 13),
      child: Center(
        child: RichText(
          text: TextSpan(
            style: const TextStyle(
              fontWeight: FontWeight.w800,
              fontSize: 30,
              letterSpacing: -1.3,
            ),
            children: const [
              TextSpan(
                text: "Say it ",
                style: TextStyle(
                  color: Colors.green,
                  shadows: [
                    Shadow(
                      color: Colors.black12,
                      offset: Offset(0, 2),
                      blurRadius: 6,
                    ),
                  ],
                ),
              ),
              TextSpan(
                text: "See it",
                style: TextStyle(
                  color: Colors.blue,
                  shadows: [
                    Shadow(
                      color: Colors.black12,
                      offset: Offset(0, 2),
                      blurRadius: 6,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xff232531),
      body: Center(
        child: SingleChildScrollView(
          child: Container(
            constraints: const BoxConstraints(maxWidth: 410),
            padding: const EdgeInsets.symmetric(vertical: 26, horizontal: 0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                _mainTitle(),
                _exampleCard(),
                _inputCard(),
                _outputCard(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
