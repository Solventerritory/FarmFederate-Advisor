import 'package:flutter/material.dart';

void main() => runApp(const FarmFederateApp());

class FarmFederateApp extends StatelessWidget {
  const FarmFederateApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FarmFederate',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.green,
        scaffoldBackgroundColor: const Color(0xFFF6F7F9),
      ),
      home: const HomeScreen(),
      routes: {
        '/hardware': (_) => const HardwareScreen(),
      },
    );
  }
}
