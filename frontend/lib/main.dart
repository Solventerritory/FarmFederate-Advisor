// lib/main.dart

import 'package:flutter/material.dart';
import 'screens/chat_screen.dart';

void main() {
  runApp(const FarmFederateApp());
}

class FarmFederateApp extends StatelessWidget {
  const FarmFederateApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FarmFederate Advisor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.purple,
        useMaterial3: true,
      ),
      home: const ChatScreen(),
    );
  }
}
