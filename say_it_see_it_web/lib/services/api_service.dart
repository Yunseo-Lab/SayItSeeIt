import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:http_parser/http_parser.dart';

class CardNewsResult {
  final bool success;
  final Uint8List? imageBytes;
  final String? statusMessage;
  CardNewsResult({required this.success, this.imageBytes, this.statusMessage});
}

class ApiService {
  static const String baseUrl = "http://localhost:7860"; // 서버 주소에 맞게 수정

  static Future<CardNewsResult> generateCardNews({
    required String query,
    List<XFile>? imageFiles,
    XFile? logoFile,
  }) async {
    final uri = Uri.parse('$baseUrl/api/cardnews');
    var request = http.MultipartRequest('POST', uri);
    request.fields['query'] = query;

    // 이미지 업로드 (웹에서도 동작하도록 fromBytes 사용)
    if (imageFiles != null) {
      for (var img in imageFiles) {
        final bytes = await img.readAsBytes();
        request.files.add(
          http.MultipartFile.fromBytes(
            'images',
            bytes,
            filename: img.name,
            contentType: _lookupMimeType(img.name),
          ),
        );
      }
    }

    // 로고 업로드 (웹에서도 동작하도록 fromBytes 사용)
    if (logoFile != null) {
      final logoBytes = await logoFile.readAsBytes();
      request.files.add(
        http.MultipartFile.fromBytes(
          'logo',
          logoBytes,
          filename: logoFile.name,
          contentType: _lookupMimeType(logoFile.name),
        ),
      );
    }

    try {
      final streamed = await request.send();
      final resp = await http.Response.fromStream(streamed);
      if (resp.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(resp.body);
        if (data.containsKey('image_base64')) {
          return CardNewsResult(
            success: true,
            imageBytes: base64Decode(data['image_base64']),
            statusMessage: data['status_message'] ?? '성공!',
          );
        } else {
          return CardNewsResult(
            success: false,
            statusMessage: data['status_message'],
          );
        }
      } else {
        return CardNewsResult(success: false, statusMessage: '서버 오류');
      }
    } catch (e) {
      return CardNewsResult(success: false, statusMessage: '네트워크 오류');
    }
  }
}

// 이미지/로고 확장자에 따른 Content-Type 자동 지정
MediaType? _lookupMimeType(String filename) {
  if (filename.endsWith('.png')) return MediaType('image', 'png');
  if (filename.endsWith('.jpg') || filename.endsWith('.jpeg'))
    return MediaType('image', 'jpeg');
  if (filename.endsWith('.gif')) return MediaType('image', 'gif');
  return null; // 기본값(null)도 괜찮음
}
