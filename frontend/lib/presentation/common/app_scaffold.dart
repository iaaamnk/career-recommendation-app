import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/auth_provider.dart';
import '../../core/theme/app_theme.dart';

class AppScaffold extends ConsumerWidget {
  final String currentRoute;
  final Widget body;

  const AppScaffold({
    super.key,
    required this.currentRoute,
    required this.body,
  });

  void _navigateTo(BuildContext context, String routeName) {
    if (currentRoute == routeName) return;
    Navigator.of(context).pushReplacementNamed(routeName);
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isDesktop = MediaQuery.of(context).size.width > 700;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        toolbarHeight: 80,
        iconTheme: const IconThemeData(color: AppTheme.primaryNavy),
        title: Padding(
          padding: const EdgeInsets.only(left: 12.0),
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
        actions: [
          if (isDesktop) ...[
            _navButton(context, 'Dashboard', '/dashboard'),
            _navButton(context, 'Assessment', '/assessment'),
            _navButton(context, 'ATS Scan', '/resume'),
            _navButton(context, 'History', '/history'),
            _navButton(context, 'Profile', '/profile'),
          ] else ...[
            PopupMenuButton<String>(
              icon: const Icon(Icons.menu, color: AppTheme.primaryNavy),
              onSelected: (val) => _navigateTo(context, val),
              itemBuilder: (context) => [
                const PopupMenuItem(value: '/dashboard', child: Text('Dashboard')),
                const PopupMenuItem(value: '/assessment', child: Text('Assessment / Quiz')),
                const PopupMenuItem(value: '/resume', child: Text('ATS Scan')),
                const PopupMenuItem(value: '/history', child: Text('History')),
                const PopupMenuItem(value: '/profile', child: Text('Profile')),
              ],
            ),
          ],
          Padding(
            padding: const EdgeInsets.only(right: 24.0, left: 16.0),
            child: IconButton(
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
          ),
        ],
      ),
      body: SafeArea(child: body),
    );
  }

  Widget _navButton(BuildContext context, String title, String routeName) {
    final isSelected = currentRoute == routeName;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8.0),
      child: TextButton(
        onPressed: () => _navigateTo(context, routeName),
        style: TextButton.styleFrom(
          foregroundColor: isSelected ? AppTheme.burntSienna : AppTheme.primaryNavy,
          textStyle: GoogleFonts.inter(
            fontWeight: isSelected ? FontWeight.w700 : FontWeight.w400,
            letterSpacing: 0.5,
          ),
        ),
        child: Text(title.toUpperCase()),
      ),
    );
  }
}
