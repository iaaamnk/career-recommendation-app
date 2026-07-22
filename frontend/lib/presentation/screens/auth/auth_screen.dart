import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../providers/auth_provider.dart';
import '../../../core/theme/app_theme.dart';

class AuthScreen extends ConsumerStatefulWidget {
  const AuthScreen({super.key});

  @override
  ConsumerState<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends ConsumerState<AuthScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isLogin = true;
  bool _isLoading = false;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    setState(() => _isLoading = true);
    final email = _emailController.text.trim();
    final password = _passwordController.text.trim();

    final authRepo = ref.read(authRepositoryProvider);

    try {
      if (_isLogin) {
        await authRepo.signIn(email: email, password: password);
      } else {
        await authRepo.signUp(email: email, password: password);
      }
      if (mounted) {
        Navigator.of(context).pushReplacementNamed('/dashboard');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(e.toString()),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDesktop = MediaQuery.of(context).size.width > 800;

    return Scaffold(
      body: Row(
        children: [
          // Left side: Editorial Hero Graphic (Desktop only)
          if (isDesktop)
            Expanded(
              child: Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    colors: [AppTheme.primaryNavy, AppTheme.softBlue],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
                padding: const EdgeInsets.all(64),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      'PathFinder.',
                      style: GoogleFonts.playfairDisplay(
                        fontSize: 64,
                        color: Colors.white,
                        fontWeight: FontWeight.w700,
                      ),
                    ).animate().fade(duration: 800.ms).slideX(begin: -0.1),
                    const SizedBox(height: 24),
                    Text(
                      'Map your career trajectory with precision AI analysis.',
                      style: GoogleFonts.inter(
                        fontSize: 24,
                        color: Colors.grey[400],
                        height: 1.5,
                      ),
                    ).animate().fade(delay: 200.ms, duration: 800.ms).slideX(begin: -0.1),
                  ],
                ),
              ),
            ),

          // Right side: Interactive Auth Form
          Expanded(
            child: Center(
              child: Container(
                constraints: const BoxConstraints(maxWidth: 400),
                padding: const EdgeInsets.all(32),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (!isDesktop) ...[
                      Text(
                        'PathFinder.',
                        style: GoogleFonts.playfairDisplay(
                          fontSize: 48,
                          fontWeight: FontWeight.bold,
                          color: AppTheme.primaryNavy,
                        ),
                      ),
                      const SizedBox(height: 48),
                    ],
                    Text(
                      _isLogin ? 'Welcome back.' : 'Begin your journey.',
                      style: Theme.of(context).textTheme.headlineMedium,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Enter your credentials to continue.',
                      style: Theme.of(context).textTheme.bodyLarge,
                    ),
                    const SizedBox(height: 48),
                    TextField(
                      controller: _emailController,
                      decoration: const InputDecoration(labelText: 'Email Address'),
                      keyboardType: TextInputType.emailAddress,
                    ),
                    const SizedBox(height: 16),
                    TextField(
                      controller: _passwordController,
                      decoration: const InputDecoration(labelText: 'Password'),
                      obscureText: true,
                    ),
                    const SizedBox(height: 32),
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton(
                        onPressed: _isLoading ? null : _submit,
                        child: _isLoading
                            ? const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  color: Colors.white,
                                  strokeWidth: 2,
                                ),
                              )
                            : Text(_isLogin ? 'SIGN IN' : 'CREATE ACCOUNT'),
                      ),
                    ),
                    const SizedBox(height: 24),
                    Center(
                      child: TextButton(
                        onPressed: () => setState(() => _isLogin = !_isLogin),
                        style: TextButton.styleFrom(
                          foregroundColor: AppTheme.primaryNavy,
                        ),
                        child: Text(
                          _isLogin
                              ? 'Need an account? Sign up'
                              : 'Already have an account? Sign in',
                        ),
                      ),
                    ),
                  ],
                ).animate().fade(duration: 600.ms).slideY(begin: 0.05),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
