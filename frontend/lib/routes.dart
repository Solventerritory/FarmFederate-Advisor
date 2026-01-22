import 'screens/login_screen.dart';
import 'screens/register_screen.dart';
import 'screens/chat_screen.dart';

import 'screens/dashboard_screen.dart';
import 'screens/ai_chat_screen.dart';
import 'screens/auth_check_screen.dart';
import 'screens/profile_screen.dart';
import 'screens/settings_screen.dart';
import 'main.dart';

final routes = {
  '/': (context) => const AuthCheckScreen(),
  '/login': (context) => const LoginScreen(),
  '/register': (context) => const RegisterScreen(),
  '/dashboard': (context) => const DashboardScreen(apiBase: 'http://localhost:8000'),
  '/home': (context) => const HomePage(),
  '/chat': (context) => const AIChatScreen(apiBase: 'http://localhost:8000'),
  '/chat-old': (context) => const ChatScreen(apiBase: 'http://localhost:8000'),

  '/profile': (context) => const ProfileScreen(),
  '/settings': (context) => const SettingsScreen(),
};
