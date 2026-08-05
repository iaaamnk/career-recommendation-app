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
  bool _isCollapsed = false;

  void _navigateTo(BuildContext context, String routeName) {
    if (widget.currentRoute == routeName) return;
    Navigator.of(context).pushReplacementNamed(routeName);
  }

  @override
  Widget build(BuildContext context) {
    final isDesktop = MediaQuery.of(context).size.width > 700;

    return Scaffold(
      body: SafeArea(
        child: Row(
          children: [
            if (!isDesktop && _isCollapsed) 
              const SizedBox.shrink()
            else
              _buildSideBar(),
            Expanded(
              child: Scaffold(
                appBar: AppBar(
                  backgroundColor: Colors.transparent,
                  elevation: 0,
                  toolbarHeight: 80,
                  leading: IconButton(
                    icon: const Icon(Icons.menu, color: AppTheme.primaryNavy),
                    onPressed: () => setState(() => _isCollapsed = !_isCollapsed),
                  ),
                  title: GestureDetector(
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
                  actions: [
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
                body: widget.body,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSideBar() {
    final user = ref.read(authStateProvider).value;
    final name = user?.name ?? 'User Account';
    final email = user?.email ?? '';

    return AnimatedContainer(
      duration: const Duration(milliseconds: 250),
      width: _isCollapsed ? 80 : 280,
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border(right: BorderSide(color: Colors.grey[200]!)),
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(16.0),
            color: AppTheme.primaryNavy,
            width: double.infinity,
            child: Column(
              children: [
                const CircleAvatar(
                  backgroundColor: Colors.white,
                  radius: 24,
                  child: Icon(Icons.person, color: AppTheme.primaryNavy, size: 30),
                ),
              ],
            ),
          ),
          if (!_isCollapsed)
            const SizedBox(height: 16),
          Expanded(
            child: ListView(
              padding: EdgeInsets.zero,
              children: [
                _navTile(Icons.dashboard, 'Dashboard', '/dashboard'),
                _navTile(Icons.assessment, 'Assessment', '/assessment'),
                _navTile(Icons.document_scanner, 'ATS Scan', '/resume'),
                _navTile(Icons.history, 'History', '/history'),
                _navTile(Icons.person, 'Profile', '/profile'),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _navTile(IconData icon, String title, String route) {
    final isSelected = widget.currentRoute == route;
    return ListTile(
      leading: Icon(icon, color: isSelected ? AppTheme.burntSienna : AppTheme.primaryNavy),
      title: _isCollapsed ? null : Text(
        title,
        style: TextStyle(
          color: isSelected ? AppTheme.burntSienna : AppTheme.primaryNavy,
          fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
        ),
      ),
      selected: isSelected,
      onTap: () => _navigateTo(context, route),
    );
  }
}
