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
      drawer: isDesktop ? null : _buildDrawer(context, ref),
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

  Widget _buildDrawer(BuildContext context, WidgetRef ref) {
    final user = ref.read(authStateProvider).value;
    final name = user?.name ?? 'User Account';
    final email = user?.email ?? '';

    return Drawer(
      child: Column(
        children: [
          UserAccountsDrawerHeader(
            decoration: const BoxDecoration(color: AppTheme.primaryNavy),
            currentAccountPicture: const CircleAvatar(
              backgroundColor: Colors.white,
              child: Icon(Icons.person, color: AppTheme.primaryNavy, size: 40),
            ),
            accountName: Text(name, style: const TextStyle(fontWeight: FontWeight.bold)),
            accountEmail: Text(email),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Profile details:', style: TextStyle(color: Colors.grey[600], fontSize: 12)),
                const SizedBox(height: 8),
                const Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [Text('Age:'), Text('24', style: TextStyle(fontWeight: FontWeight.bold))],
                ),
                const SizedBox(height: 4),
                const Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [Text('Skills:'), Text('Python, SQL', style: TextStyle(fontWeight: FontWeight.bold))],
                ),
                const SizedBox(height: 4),
                const Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [Text('Interests:'), Text('Data, Analysis', style: TextStyle(fontWeight: FontWeight.bold))],
                ),
                const SizedBox(height: 4),
                const Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [Text('RIASEC:'), Text('R:5 I:5 A:5 S:5 E:5 C:5', style: TextStyle(fontWeight: FontWeight.bold))],
                ),
              ],
            ),
          ),
          const Divider(),
          Expanded(
            child: ListView(
              padding: EdgeInsets.zero,
              children: [
                ListTile(
                  leading: const Icon(Icons.dashboard),
                  title: const Text('Dashboard'),
                  selected: currentRoute == '/dashboard',
                  onTap: () {
                    Navigator.pop(context);
                    _navigateTo(context, '/dashboard');
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.assessment),
                  title: const Text('Assessment / Quiz'),
                  selected: currentRoute == '/assessment',
                  onTap: () {
                    Navigator.pop(context);
                    _navigateTo(context, '/assessment');
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.document_scanner),
                  title: const Text('ATS Scan'),
                  selected: currentRoute == '/resume',
                  onTap: () {
                    Navigator.pop(context);
                    _navigateTo(context, '/resume');
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.history),
                  title: const Text('History'),
                  selected: currentRoute == '/history',
                  onTap: () {
                    Navigator.pop(context);
                    _navigateTo(context, '/history');
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.person),
                  title: const Text('Profile'),
                  selected: currentRoute == '/profile',
                  onTap: () {
                    Navigator.pop(context);
                    _navigateTo(context, '/profile');
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
