import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'widgets/chat_screen.dart';
import 'services/chat_state.dart';

void main() {
  runApp(const FarmFederateApp());
}

class FarmFederateApp extends StatelessWidget {
  const FarmFederateApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => ChatState(),
      child: MaterialApp(
        title: 'FarmFederate Chat',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          primarySwatch: Colors.green,
          scaffoldBackgroundColor: Colors.grey[50],
        ),
        home: const ChatScreen(),
      ),
    );
  }
}
