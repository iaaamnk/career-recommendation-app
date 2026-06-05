import re

with open("lib/main.dart", "r") as f:
    content = f.read()

# 1. Imports
imports = """import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter/foundation.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'firebase_options.dart';
"""
content = re.sub(r"import 'package:flutter/material\.dart';.*import 'package:url_launcher/url_launcher\.dart';", imports, content, flags=re.DOTALL)

# 2. Main
main_func = """void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  runApp(const ProviderScope(child: PathFinderApp()));
}

final authStateProvider = StreamProvider<User?>((ref) {
  return FirebaseAuth.instance.authStateChanges();
});
"""
content = re.sub(r"void main\(\) \{\n  runApp\(const ProviderScope\(child: PathFinderApp\(\)\)\);\n\}", main_func, content)

# 3. PathFinderApp home
auth_wrapper = """      home: const AuthWrapper(),
    );
  }
}

class AuthWrapper extends ConsumerWidget {
  const AuthWrapper({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final authState = ref.watch(authStateProvider);
    return authState.when(
      data: (user) {
        if (user == null) return const AuthScreen();
        return const MainLayout();
      },
      loading: () => const Scaffold(body: Center(child: CircularProgressIndicator())),
      error: (e, trace) => Scaffold(body: Center(child: Text('Error: $e'))),
    );
  }
}

class AuthScreen extends StatefulWidget {
  const AuthScreen({super.key});

  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isLogin = true;
  bool _isLoading = false;

  Future<void> _submit() async {
    setState(() => _isLoading = true);
    try {
      if (_isLogin) {
        await FirebaseAuth.instance.signInWithEmailAndPassword(
          email: _emailController.text.trim(),
          password: _passwordController.text.trim(),
        );
      } else {
        await FirebaseAuth.instance.createUserWithEmailAndPassword(
          email: _emailController.text.trim(),
          password: _passwordController.text.trim(),
        );
      }
    } on FirebaseAuthException catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message ?? 'Authentication error')));
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('PathFinder AI Login')),
      body: Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 400),
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.explore, size: 64, color: Color(0xFF6C63FF)),
              const SizedBox(height: 32),
              TextField(
                controller: _emailController,
                decoration: const InputDecoration(labelText: 'Email', border: OutlineInputBorder()),
                keyboardType: TextInputType.emailAddress,
              ),
              const SizedBox(height: 16),
              TextField(
                controller: _passwordController,
                decoration: const InputDecoration(labelText: 'Password', border: OutlineInputBorder()),
                obscureText: true,
              ),
              const SizedBox(height: 24),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: _isLoading ? null : _submit,
                  style: FilledButton.styleFrom(padding: const EdgeInsets.all(16)),
                  child: _isLoading ? const CircularProgressIndicator(color: Colors.white) : Text(_isLogin ? 'Login' : 'Sign Up'),
                ),
              ),
              const SizedBox(height: 16),
              TextButton(
                onPressed: () => setState(() => _isLogin = !_isLogin),
                child: Text(_isLogin ? 'Create an account' : 'I already have an account'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
"""
content = re.sub(r"      home: const MainLayout\(\),\n    \);\n  }\n}", auth_wrapper, content)

# 4. Update MainLayout to include Logout
logout_btn = """          TextButton(
            onPressed: () => setState(() => currentView = 'profile'),
            child: const Text('Profile'),
          ),
          IconButton(
            onPressed: () => FirebaseAuth.instance.signOut(),
            icon: const Icon(Icons.logout),
            tooltip: 'Logout',
          ),
          const SizedBox(width: 8),"""
content = re.sub(r"          TextButton\(\n            onPressed: \(\) => setState\(\(\) => currentView = 'profile'\),\n            child: const Text\('Profile'\),\n          \),\n          const SizedBox\(width: 8\),", logout_btn, content)


# 5. Assessment POST Request
old_assess_post = """    try {
      final response = await http.post(
        Uri.parse('https://career-recommendation-app-2-08ny.onrender.com/api/recommend'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          "user_id": 1, // Mock user ID"""
new_assess_post = """    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      setState(() => isLoading = false);
      return;
    }
    final token = await user.getIdToken();

    try {
      final response = await http.post(
        Uri.parse('https://career-recommendation-app-2-08ny.onrender.com/api/recommend'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $token',
        },
        body: jsonEncode({"""
content = content.replace(old_assess_post, new_assess_post)

# 6. Resume POST Request
old_resume_post = """    try {
      final response = await http.post(
        Uri.parse('https://career-recommendation-app-2-08ny.onrender.com/api/resume/analyze'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          "user_id": 1, // Mock user ID"""
new_resume_post = """    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      setState(() => isLoading = false);
      return;
    }
    final token = await user.getIdToken();

    try {
      final response = await http.post(
        Uri.parse('https://career-recommendation-app-2-08ny.onrender.com/api/resume/analyze'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $token',
        },
        body: jsonEncode({"""
content = content.replace(old_resume_post, new_resume_post)

# Update ProfileView to show actual user info
old_profile = """  final _nameController = TextEditingController(text: 'Demo User');
  final _emailController = TextEditingController(text: 'demo@example.com');"""
new_profile = """  final _nameController = TextEditingController(text: FirebaseAuth.instance.currentUser?.displayName ?? '');
  final _emailController = TextEditingController(text: FirebaseAuth.instance.currentUser?.email ?? '');"""
content = content.replace(old_profile, new_profile)

with open("lib/main.dart", "w") as f:
    f.write(content)
