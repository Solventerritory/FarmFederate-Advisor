import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import '../services/api_service.dart';
import '../widgets/result_card.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  XFile? _image;
  bool _loading = false;
  String _text = '';
  Map<String, double> _probs = {};

  final List<String> classes = [
    'water_stress',
    'nutrient_def',
    'pest_risk',
    'disease_risk',
    'heat_stress'
  ];

  Future<void> _pickImage(ImageSource src) async {
    final XFile? img = await _picker.pickImage(source: src, imageQuality: 80);
    if (img == null) return;
    setState(() {
      _image = img;
      _probs = {};
    });
  }

  Future<void> _predict() async {
    if (_image == null && _text.trim().isEmpty) {
      _showSnack('Provide an image or text to run inference.');
      return;
    }
    setState(() {
      _loading = true;
      _probs = {};
    });
    try {
      final res = await ApiService.predict(File(_image?.path ?? ''), _text);
      final classesRes = List<String>.from(res['classes']);
      final probsList = List<dynamic>.from(res['probs']);
      Map<String, double> out = {};
      for (var i = 0; i < classesRes.length; i++) {
        out[classesRes[i]] = (probsList[i] as num).toDouble();
      }
      setState(() { _probs = out; });
    } catch (e) {
      _showSnack('Failed to predict: $e');
    } finally { setState(() => _loading = false); }
  }

  void _showSnack(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  void _showPickDialog() {
    showModalBottomSheet(
      context: context,
      builder: (_) => SafeArea(
        child: Wrap(
          children: [
            ListTile(
              leading: const Icon(Icons.photo_library),
              title: const Text('Pick from gallery'),
              onTap: () { Navigator.pop(context); _pickImage(ImageSource.gallery); },
            ),
            ListTile(
              leading: const Icon(Icons.camera_alt),
              title: const Text('Take a photo'),
              onTap: () { Navigator.pop(context); _pickImage(ImageSource.camera); },
            ),
            if (_image != null)
              ListTile(
                leading: const Icon(Icons.delete_forever),
                title: const Text('Remove image'),
                onTap: () { Navigator.pop(context); setState(() { _image = null; _probs = {}; }); },
              ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('FarmFederate — Crop Stress Detector'),
        backgroundColor: Colors.transparent,
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.hardware_outlined),
            onPressed: () => Navigator.pushNamed(context, '/hardware'),
            tooltip: 'Hardware Dashboard',
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          child: Column(
            children: [
              Card(
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                elevation: 4,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    children: [
                      GestureDetector(
                        onTap: _showPickDialog,
                        child: Container(
                          height: 220,
                          width: double.infinity,
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(8),
                            color: Colors.grey.shade100,
                            border: Border.all(color: Colors.grey.shade300),
                          ),
                          child: _image == null
                              ? Center(
                                  child: Column(
                                    mainAxisSize: MainAxisSize.min,
                                    children: const [
                                      Icon(Icons.image, size: 48, color: Colors.grey),
                                      SizedBox(height: 8),
                                      Text('Tap to pick or take a photo', style: TextStyle(color: Colors.grey)),
                                    ],
                                  ),
                                )
                              : ClipRRect(
                                  borderRadius: BorderRadius.circular(8),
                                  child: Image.file(File(_image!.path), fit: BoxFit.cover),
                                ),
                        ),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        maxLines: 3,
                        decoration: const InputDecoration(
                          hintText: 'Describe the field / symptoms (optional)',
                          border: OutlineInputBorder(),
                        ),
                        onChanged: (v) => _text = v,
                      ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Expanded(
                            child: ElevatedButton.icon(
                              onPressed: _loading ? null : _predict,
                              icon: const Icon(Icons.analytics_outlined),
                              label: Text(_loading ? 'Running...' : 'Analyze'),
                              style: ElevatedButton.styleFrom(
                                padding: const EdgeInsets.symmetric(vertical: 14),
                                backgroundColor: Colors.green.shade700,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Expanded(
                child: _loading
                    ? const Center(child: SpinKitPulse(color: Colors.green, size: 80.0))
                    : _probs.isEmpty
                        ? Center(
                            child: Column(mainAxisSize: MainAxisSize.min, children: const [
                              Icon(Icons.info_outline, size: 40, color: Colors.grey),
                              SizedBox(height: 8),
                              Text('No results yet — run an analysis', style: TextStyle(color: Colors.grey)),
                            ]),
                          )
                        : ListView(children: _probs.entries.map((e) => ResultCard(label: e.key, prob: e.value)).toList()),
              ),
            ],
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _showPickDialog,
        backgroundColor: Colors.green,
        child: const Icon(Icons.add_a_photo),
      ),
    );
  }
}
