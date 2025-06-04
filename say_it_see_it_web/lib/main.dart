import 'package:flutter/material.dart';
import 'pages/home_page.dart';

void main() {
  runApp(const SayItSeeItApp());
}

class SayItSeeItApp extends StatelessWidget {
  const SayItSeeItApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '카드뉴스 생성기',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: HomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}
