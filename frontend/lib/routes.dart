import 'screens/login_screen.dart';
import 'screens/register_screen.dart';
import 'screens/chat_screen.dart';

import 'screens/dashboard_screen.dart';
import 'screens/ai_chat_screen.dart';
import 'screens/auth_check_screen.dart';
import 'screens/profile_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/qdrant_demo_screen.dart';
import 'constants.dart';
import 'main.dart';

final routes = {
  '/': (context) => const AuthCheckScreen(),
  '/login': (context) => const LoginScreen(),
  '/register': (context) => const RegisterScreen(),
  '/dashboard': (context) => const DashboardScreen(apiBase: DEFAULT_BACKEND),
  '/home': (context) => const HomePage(),
  '/chat': (context) => const AIChatScreen(apiBase: DEFAULT_BACKEND),
  '/chat-old': (context) => const ChatScreen(apiBase: DEFAULT_BACKEND),
  '/qdrant': (context) => const QdrantDemoScreen(apiBase: DEFAULT_BACKEND),
  '/profile': (context) => const ProfileScreen(),
  '/settings': (context) => const SettingsScreen(),
};
