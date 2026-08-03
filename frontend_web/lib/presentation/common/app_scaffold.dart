import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/auth_provider.dart';
import '../../core/theme/app_theme.dart';

class AppScaffold extends ConsumerStatefulWidget {
  final String currentRoute;
  final Widget body;

  const AppScaffold({
    super.key,
    required this.currentRoute,
    required this.body,
  });

  @override
  ConsumerState<AppScaffold> createState() => _AppScaffoldState();
}

class _AppScaffoldState extends ConsumerState<AppScaffold> {
  void _navigateTo(BuildContext context, String routeName) {
    if (widget.currentRoute == routeName) return;
    Navigator.of(context).pushReplacementNamed(routeName);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F9FA), // Slightly off-white for premium feel
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(80),
        child: Container(
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.9),
            border: Border(bottom: BorderSide(color: Colors.grey.withValues(alpha: 0.1))),
          ),
          child: SafeArea(
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1200),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      MouseRegion(
                        cursor: SystemMouseCursors.click,
                        child: GestureDetector(
                          onTap: () => _navigateTo(context, '/dashboard'),
                          child: Text(
                            'PathFinder.',
                            style: GoogleFonts.playfairDisplay(
                              fontSize: 28,
                              fontWeight: FontWeight.bold,
                              color: AppTheme.primaryNavy,
                            ),
                          ),
                        ),
                      ),
                      Row(
                        children: [
                          _navButton(context, 'Dashboard', '/dashboard'),
                          _navButton(context, 'Assessment', '/assessment'),
                          _navButton(context, 'ATS Scan', '/resume'),
                          _navButton(context, 'History', '/history'),
                          _navButton(context, 'Profile', '/profile'),
                          const SizedBox(width: 24),
                          Container(
                            height: 32,
                            width: 1,
                            color: Colors.grey.withValues(alpha: 0.3),
                          ),
                          const SizedBox(width: 16),
                          IconButton(
                            onPressed: () async {
                              final authRepo = ref.read(authRepositoryProvider);
                              await authRepo.signOut();
                              if (context.mounted) {
                                Navigator.of(context).pushReplacementNamed('/auth');
                              }
                            },
                            icon: const Icon(Icons.logout, color: AppTheme.primaryNavy),
                            tooltip: 'Sign Out',
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                children: [
                  Center(
                    child: ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 1200),
                      child: widget.body,
                    ),
                  ),
                  const _WebFooter(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _navButton(BuildContext context, String title, String routeName) {
    final isSelected = widget.currentRoute == routeName;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12.0),
      child: TextButton(
        onPressed: () => _navigateTo(context, routeName),
        style: TextButton.styleFrom(
          foregroundColor: isSelected ? AppTheme.burntSienna : AppTheme.primaryNavy,
          textStyle: GoogleFonts.inter(
            fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
            fontSize: 15,
            letterSpacing: 0.5,
          ),
        ),
        child: Text(title),
      ),
    );
  }
}

class _WebFooter extends StatelessWidget {
  const _WebFooter();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      color: AppTheme.primaryNavy,
      padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 1200),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'PathFinder.',
                    style: GoogleFonts.playfairDisplay(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'AI-driven career trajectory optimization.',
                    style: GoogleFonts.inter(color: Colors.white70),
                  ),
                ],
              ),
              Row(
                children: [
                  TextButton(
                    onPressed: () {},
                    child: Text('Privacy Policy', style: GoogleFonts.inter(color: Colors.white70)),
                  ),
                  const SizedBox(width: 24),
                  TextButton(
                    onPressed: () {},
                    child: Text('Terms of Service', style: GoogleFonts.inter(color: Colors.white70)),
                  ),
                ],
              )
            ],
          ),
        ),
      ),
    );
  }
}
