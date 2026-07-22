import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../common/app_scaffold.dart';
import '../../providers/auth_provider.dart';
import '../../../core/theme/app_theme.dart';

class ProfileScreen extends ConsumerStatefulWidget {
  const ProfileScreen({super.key});

  @override
  ConsumerState<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends ConsumerState<ProfileScreen> {
  late TextEditingController _nameController;
  late TextEditingController _emailController;

  @override
  void initState() {
    super.initState();
    final user = ref.read(authStateProvider).value;
    _nameController = TextEditingController(text: user?.name ?? '');
    _emailController = TextEditingController(text: user?.email ?? '');
  }

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AppScaffold(
      currentRoute: '/profile',
      body: Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 800),
          padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 48.0),
          child: ListView(
            children: [
              Text('User Profile.', style: Theme.of(context).textTheme.displayMedium)
                  .animate()
                  .fade()
                  .slideY(begin: 0.1),
              const SizedBox(height: 16),
              Text(
                'Manage your personal account credentials and profile details.',
                style: Theme.of(context)
                    .textTheme
                    .bodyLarge
                    ?.copyWith(fontSize: 18),
              ).animate().fade(delay: 100.ms).slideY(begin: 0.1),
              const SizedBox(height: 48),

              Container(
                color: Colors.white,
                padding: const EdgeInsets.all(48),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const CircleAvatar(
                          radius: 36,
                          backgroundColor: AppTheme.primaryNavy,
                          child: Icon(Icons.person, size: 40, color: Colors.white),
                        ),
                        const SizedBox(width: 24),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                _nameController.text.isEmpty
                                    ? 'User Account'
                                    : _nameController.text,
                                style: Theme.of(context).textTheme.titleLarge?.copyWith(fontSize: 22),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                _emailController.text,
                                style: TextStyle(color: Colors.grey[600]),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 48),
                    const Divider(),
                    const SizedBox(height: 48),

                    Text(
                      'ACCOUNT DETAILS',
                      style: GoogleFonts.inter(
                        color: Colors.grey[500],
                        fontWeight: FontWeight.w700,
                        letterSpacing: 2,
                        fontSize: 12,
                      ),
                    ),
                    const SizedBox(height: 32),
                    TextField(
                      controller: _nameController,
                      decoration: const InputDecoration(labelText: 'Full Name'),
                    ),
                    const SizedBox(height: 24),
                    TextField(
                      controller: _emailController,
                      decoration: const InputDecoration(labelText: 'Email Address'),
                      keyboardType: TextInputType.emailAddress,
                      enabled: false,
                    ),
                    const SizedBox(height: 48),
                    FilledButton(
                      onPressed: () {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text('Profile saved successfully.'),
                            backgroundColor: AppTheme.primaryNavy,
                          ),
                        );
                      },
                      child: const Text('SAVE CHANGES'),
                    ),
                  ],
                ),
              ).animate().fade(delay: 200.ms).slideY(begin: 0.1),
            ],
          ),
        ),
      ),
    );
  }
}
