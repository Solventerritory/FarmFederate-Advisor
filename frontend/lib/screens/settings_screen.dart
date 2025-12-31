import 'package:flutter/material.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({Key? key}) : super(key: key);

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _notificationsEnabled = true;
  bool _emailNotifications = true;
  bool _pushNotifications = false;
  bool _darkMode = true;
  bool _autoRefresh = true;
  bool _soundEffects = false;
  String _language = 'English';
  String _temperatureUnit = 'Celsius';
  String _dataRefreshRate = '5 seconds';
  double _alertThreshold = 80.0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E21),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1D1E33),
        title: const Text('Settings'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Appearance Section
          _buildSectionTitle('Appearance'),
          const SizedBox(height: 12),
          _buildSettingCard(
            icon: Icons.dark_mode,
            title: 'Dark Mode',
            subtitle: 'Use dark theme throughout the app',
            trailing: Switch(
              value: _darkMode,
              onChanged: (value) {
                setState(() {
                  _darkMode = value;
                });
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Theme will be applied on restart')),
                );
              },
              activeThumbColor: const Color(0xFF1D976C),
            ),
          ),
          const SizedBox(height: 12),
          _buildSettingCard(
            icon: Icons.language,
            title: 'Language',
            subtitle: _language,
            trailing: const Icon(Icons.arrow_forward_ios, size: 16, color: Colors.white60),
            onTap: () => _showLanguageDialog(),
          ),
          const SizedBox(height: 30),

          // Notifications Section
          _buildSectionTitle('Notifications'),
          const SizedBox(height: 12),
          _buildSettingCard(
            icon: Icons.notifications,
            title: 'All Notifications',
            subtitle: 'Enable or disable all notifications',
            trailing: Switch(
              value: _notificationsEnabled,
              onChanged: (value) {
                setState(() {
                  _notificationsEnabled = value;
                  if (!value) {
                    _emailNotifications = false;
                    _pushNotifications = false;
                  }
                });
              },
              activeThumbColor: const Color(0xFF1D976C),
            ),
          ),
          const SizedBox(height: 12),
          _buildSettingCard(
            icon: Icons.email,
            title: 'Email Notifications',
            subtitle: 'Receive alerts via email',
            trailing: Switch(
              value: _emailNotifications,
              onChanged: _notificationsEnabled
                  ? (value) {
                      setState(() {
                        _emailNotifications = value;
                      });
                    }
                  : null,
              activeThumbColor: const Color(0xFF1D976C),
            ),
          ),
          const SizedBox(height: 12),
          _buildSettingCard(
            icon: Icons.notification_important,
            title: 'Push Notifications',
            subtitle: 'Receive instant alerts',
            trailing: Switch(
              value: _pushNotifications,
              onChanged: _notificationsEnabled
                  ? (value) {
                      setState(() {
                        _pushNotifications = value;
                      });
                    }
                  : null,
              activeThumbColor: const Color(0xFF1D976C),
            ),
          ),
          const SizedBox(height: 30),

          // Data & Sync Section
          _buildSectionTitle('Data & Sync'),
          const SizedBox(height: 12),
          _buildSettingCard(
            icon: Icons.refresh,
            title: 'Auto Refresh',
            subtitle: 'Automatically refresh sensor data',
            trailing: Switch(
              value: _autoRefresh,
              onChanged: (value) {
                setState(() {
                  _autoRefresh = value;
                });
              },
              activeThumbColor: const Color(0xFF1D976C),
            ),
          ),
          const SizedBox(height: 12),
          _buildSettingCard(
            icon: Icons.timer,
            title: 'Data Refresh Rate',
            subtitle: _dataRefreshRate,
            trailing: const Icon(Icons.arrow_forward_ios, size: 16, color: Colors.white60),
            onTap: () => _showRefreshRateDialog(),
          ),
          const SizedBox(height: 30),

          // Units Section
          _buildSectionTitle('Units'),
          const SizedBox(height: 12),
          _buildSettingCard(
            icon: Icons.thermostat,
            title: 'Temperature Unit',
            subtitle: _temperatureUnit,
            trailing: const Icon(Icons.arrow_forward_ios, size: 16, color: Colors.white60),
            onTap: () => _showTemperatureUnitDialog(),
          ),
          const SizedBox(height: 30),

          // Alerts Section
          _buildSectionTitle('Alerts'),
          const SizedBox(height: 12),
          _buildSettingCard(
            icon: Icons.volume_up,
            title: 'Sound Effects',
            subtitle: 'Play sounds for alerts',
            trailing: Switch(
              value: _soundEffects,
              onChanged: (value) {
                setState(() {
                  _soundEffects = value;
                });
              },
              activeThumbColor: const Color(0xFF1D976C),
            ),
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: const Color(0xFF1D1E33),
              borderRadius: BorderRadius.circular(12),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.2),
                  blurRadius: 8,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Icon(Icons.warning_amber, color: Color(0xFFFFA726)),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Alert Threshold',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                          SizedBox(height: 4),
                          Text(
                            'Critical level for moisture/temperature',
                            style: TextStyle(
                              color: Colors.white60,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    ),
                    Text(
                      '${_alertThreshold.toInt()}%',
                      style: const TextStyle(
                        color: Color(0xFF93F9B9),
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                Slider(
                  value: _alertThreshold,
                  min: 50,
                  max: 100,
                  divisions: 10,
                  label: '${_alertThreshold.toInt()}%',
                  onChanged: (value) {
                    setState(() {
                      _alertThreshold = value;
                    });
                  },
                  activeColor: const Color(0xFF1D976C),
                  inactiveColor: Colors.white24,
                ),
              ],
            ),
          ),
          const SizedBox(height: 30),

          // Advanced Section
          _buildSectionTitle('Advanced'),
          const SizedBox(height: 12),
          _buildSettingCard(
            icon: Icons.storage,
            title: 'Clear Cache',
            subtitle: 'Free up storage space',
            trailing: const Icon(Icons.arrow_forward_ios, size: 16, color: Colors.white60),
            onTap: () => _showClearCacheDialog(),
          ),
          const SizedBox(height: 12),
          _buildSettingCard(
            icon: Icons.cloud_download,
            title: 'Export Data',
            subtitle: 'Download your farm data',
            trailing: const Icon(Icons.arrow_forward_ios, size: 16, color: Colors.white60),
            onTap: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Export feature coming soon')),
              );
            },
          ),
          const SizedBox(height: 12),
          _buildSettingCard(
            icon: Icons.security,
            title: 'Privacy Policy',
            subtitle: 'View our privacy policy',
            trailing: const Icon(Icons.arrow_forward_ios, size: 16, color: Colors.white60),
            onTap: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Opening privacy policy...')),
              );
            },
          ),
          const SizedBox(height: 12),
          _buildSettingCard(
            icon: Icons.info,
            title: 'About',
            subtitle: 'App version 1.0.0',
            trailing: const Icon(Icons.arrow_forward_ios, size: 16, color: Colors.white60),
            onTap: () => _showAboutDialog(),
          ),
          const SizedBox(height: 30),

          // Danger Zone
          _buildSectionTitle('Danger Zone'),
          const SizedBox(height: 12),
          _buildSettingCard(
            icon: Icons.delete_forever,
            title: 'Delete Account',
            subtitle: 'Permanently delete your account',
            trailing: const Icon(Icons.arrow_forward_ios, size: 16, color: Colors.red),
            onTap: () => _showDeleteAccountDialog(),
            titleColor: Colors.red,
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Text(
      title,
      style: const TextStyle(
        color: Colors.white,
        fontSize: 18,
        fontWeight: FontWeight.bold,
      ),
    );
  }

  Widget _buildSettingCard({
    required IconData icon,
    required String title,
    required String subtitle,
    required Widget trailing,
    VoidCallback? onTap,
    Color? titleColor,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: const Color(0xFF1D1E33),
          borderRadius: BorderRadius.circular(12),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.2),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Row(
          children: [
            Icon(icon, color: const Color(0xFF93F9B9)),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      color: titleColor ?? Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    subtitle,
                    style: const TextStyle(
                      color: Colors.white60,
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ),
            trailing,
          ],
        ),
      ),
    );
  }

  Future<void> _showLanguageDialog() async {
    return showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1D1E33),
        title: const Text('Select Language', style: TextStyle(color: Colors.white)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildLanguageOption('English'),
            _buildLanguageOption('Spanish'),
            _buildLanguageOption('French'),
            _buildLanguageOption('Hindi'),
            _buildLanguageOption('Mandarin'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel', style: TextStyle(color: Colors.white60)),
          ),
        ],
      ),
    );
  }

  Widget _buildLanguageOption(String language) {
    return RadioListTile<String>(
      title: Text(language, style: const TextStyle(color: Colors.white)),
      value: language,
      groupValue: _language,
      activeColor: const Color(0xFF1D976C),
      onChanged: (value) {
        setState(() {
          _language = value!;
        });
        Navigator.pop(context);
      },
    );
  }

  Future<void> _showRefreshRateDialog() async {
    return showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1D1E33),
        title: const Text('Data Refresh Rate', style: TextStyle(color: Colors.white)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildRefreshRateOption('2 seconds'),
            _buildRefreshRateOption('5 seconds'),
            _buildRefreshRateOption('10 seconds'),
            _buildRefreshRateOption('30 seconds'),
            _buildRefreshRateOption('1 minute'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel', style: TextStyle(color: Colors.white60)),
          ),
        ],
      ),
    );
  }

  Widget _buildRefreshRateOption(String rate) {
    return RadioListTile<String>(
      title: Text(rate, style: const TextStyle(color: Colors.white)),
      value: rate,
      groupValue: _dataRefreshRate,
      activeColor: const Color(0xFF1D976C),
      onChanged: (value) {
        setState(() {
          _dataRefreshRate = value!;
        });
        Navigator.pop(context);
      },
    );
  }

  Future<void> _showTemperatureUnitDialog() async {
    return showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1D1E33),
        title: const Text('Temperature Unit', style: TextStyle(color: Colors.white)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            RadioListTile<String>(
              title: const Text('Celsius (°C)', style: TextStyle(color: Colors.white)),
              value: 'Celsius',
              groupValue: _temperatureUnit,
              activeColor: const Color(0xFF1D976C),
              onChanged: (value) {
                setState(() {
                  _temperatureUnit = value!;
                });
                Navigator.pop(context);
              },
            ),
            RadioListTile<String>(
              title: const Text('Fahrenheit (°F)', style: TextStyle(color: Colors.white)),
              value: 'Fahrenheit',
              groupValue: _temperatureUnit,
              activeColor: const Color(0xFF1D976C),
              onChanged: (value) {
                setState(() {
                  _temperatureUnit = value!;
                });
                Navigator.pop(context);
              },
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel', style: TextStyle(color: Colors.white60)),
          ),
        ],
      ),
    );
  }

  Future<void> _showClearCacheDialog() async {
    return showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1D1E33),
        title: const Text('Clear Cache', style: TextStyle(color: Colors.white)),
        content: const Text(
          'This will clear all cached data. The app may load slower the next time you use it.',
          style: TextStyle(color: Colors.white70),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel', style: TextStyle(color: Colors.white60)),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Cache cleared successfully')),
              );
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF1D976C),
            ),
            child: const Text('Clear'),
          ),
        ],
      ),
    );
  }

  Future<void> _showAboutDialog() async {
    return showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1D1E33),
        title: const Text('About FarmFederate', style: TextStyle(color: Colors.white)),
        content: const Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Version: 1.0.0',
              style: TextStyle(color: Colors.white70),
            ),
            SizedBox(height: 8),
            Text(
              'A privacy-preserving AI-powered farm advisory system using federated learning.',
              style: TextStyle(color: Colors.white70),
            ),
            SizedBox(height: 16),
            Text(
              '© 2025 FarmFederate Advisor',
              style: TextStyle(color: Colors.white60, fontSize: 12),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close', style: TextStyle(color: Color(0xFF1D976C))),
          ),
        ],
      ),
    );
  }

  Future<void> _showDeleteAccountDialog() async {
    return showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1D1E33),
        title: const Text(
          'Delete Account',
          style: TextStyle(color: Colors.red),
        ),
        content: const Text(
          'This action cannot be undone. All your data will be permanently deleted.',
          style: TextStyle(color: Colors.white70),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel', style: TextStyle(color: Colors.white60)),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Account deletion requires email verification'),
                  backgroundColor: Colors.red,
                ),
              );
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red,
            ),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }
}
