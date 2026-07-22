import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../common/app_scaffold.dart';
import '../../common/metric_card.dart';
import '../../providers/auth_provider.dart';
import '../../providers/history_provider.dart';
import '../../../core/theme/app_theme.dart';

class DashboardScreen extends ConsumerWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final user = ref.watch(authStateProvider).value;
    final userName = user?.name ?? 'User';
    final historyAsync = ref.watch(historyFutureProvider);

    return AppScaffold(
      currentRoute: '/dashboard',
      body: Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 1000),
          padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 48.0),
          child: ListView(
            children: [
              Text('Overview.', style: Theme.of(context).textTheme.displayMedium)
                  .animate()
                  .fade()
                  .slideY(begin: 0.1),
              const SizedBox(height: 16),
              Text(
                'Welcome back, $userName. Here is your trajectory.',
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(fontSize: 18),
              ).animate().fade(delay: 100.ms).slideY(begin: 0.1),
              const SizedBox(height: 48),

              // Metrics Overview Section
              historyAsync.when(
                data: (history) {
                  final assessmentsTaken = history?.assessments.length ?? 0;
                  final topMatch = (history != null && history.assessments.isNotEmpty)
                      ? history.assessments.first.recommendedCareer
                      : 'None';
                  final atsScore = (history != null && history.resumes.isNotEmpty)
                      ? '${history.resumes.first.atsScore}%'
                      : 'N/A';

                  return LayoutBuilder(builder: (context, constraints) {
                    final isWide = constraints.maxWidth > 600;
                    return Flex(
                      direction: isWide ? Axis.horizontal : Axis.vertical,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(
                          flex: isWide ? 1 : 0,
                          child: MetricCard(
                            title: 'Assessments',
                            value: assessmentsTaken.toString(),
                            icon: Icons.psychology,
                            onTap: () => Navigator.of(context).pushReplacementNamed('/assessment'),
                          ),
                        ),
                        if (isWide) const SizedBox(width: 32) else const SizedBox(height: 32),
                        Expanded(
                          flex: isWide ? 1 : 0,
                          child: MetricCard(
                            title: 'Top Match',
                            value: topMatch,
                            icon: Icons.star_outline,
                            onTap: () => Navigator.of(context).pushReplacementNamed('/history'),
                          ),
                        ),
                        if (isWide) const SizedBox(width: 32) else const SizedBox(height: 32),
                        Expanded(
                          flex: isWide ? 1 : 0,
                          child: MetricCard(
                            title: 'ATS Score',
                            value: atsScore,
                            icon: Icons.document_scanner_outlined,
                            onTap: () => Navigator.of(context).pushReplacementNamed('/resume'),
                          ),
                        ),
                      ],
                    );
                  });
                },
                loading: () => const Center(
                  child: Padding(
                    padding: EdgeInsets.all(32.0),
                    child: CircularProgressIndicator(color: AppTheme.primaryNavy),
                  ),
                ),
                error: (_, stack) => LayoutBuilder(builder: (context, constraints) {
                  final isWide = constraints.maxWidth > 600;
                  return Flex(
                    direction: isWide ? Axis.horizontal : Axis.vertical,
                    children: const [
                      Expanded(child: MetricCard(title: 'Assessments', value: '0')),
                      SizedBox(width: 16, height: 16),
                      Expanded(child: MetricCard(title: 'Top Match', value: 'None')),
                      SizedBox(width: 16, height: 16),
                      Expanded(child: MetricCard(title: 'ATS Score', value: 'N/A')),
                    ],
                  );
                }),
              ).animate().fade(delay: 200.ms).slideY(begin: 0.1),

              const SizedBox(height: 64),

              // Quick Actions Banner
              Text('Quick Actions.', style: Theme.of(context).textTheme.headlineMedium)
                  .animate()
                  .fade(delay: 250.ms),
              const SizedBox(height: 24),
              LayoutBuilder(
                builder: (context, constraints) {
                  final isWide = constraints.maxWidth > 700;
                  final actionCards = [
                    _QuickActionTile(
                      title: 'Take Career Quiz',
                      subtitle: 'Discover personalized career trajectories using RIASEC scoring.',
                      icon: Icons.quiz_outlined,
                      buttonText: 'START QUIZ',
                      onTap: () => Navigator.of(context).pushReplacementNamed('/assessment'),
                    ),
                    _QuickActionTile(
                      title: 'ATS Resume Scan',
                      subtitle: 'Analyze resume match percentage, skill gaps, and interview prep.',
                      icon: Icons.description_outlined,
                      buttonText: 'SCAN RESUME',
                      onTap: () => Navigator.of(context).pushReplacementNamed('/resume'),
                    ),
                    _QuickActionTile(
                      title: 'User Profile',
                      subtitle: 'Manage your account credentials and personal details.',
                      icon: Icons.person_outline,
                      buttonText: 'EDIT PROFILE',
                      onTap: () => Navigator.of(context).pushReplacementNamed('/profile'),
                    ),
                  ];

                  if (isWide) {
                    return Row(
                      children: actionCards
                          .map((c) => Expanded(
                                child: Padding(
                                  padding: const EdgeInsets.symmetric(horizontal: 8.0),
                                  child: c,
                                ),
                              ))
                          .toList(),
                    );
                  } else {
                    return Column(
                      children: actionCards
                          .map((c) => Padding(
                                padding: const EdgeInsets.only(bottom: 16.0),
                                child: c,
                              ))
                          .toList(),
                    );
                  }
                },
              ).animate().fade(delay: 300.ms).slideY(begin: 0.05),

              const SizedBox(height: 64),

              // Recent Activity List
              Text('Recent Activity.', style: Theme.of(context).textTheme.headlineMedium)
                  .animate()
                  .fade(delay: 350.ms),
              const SizedBox(height: 24),
              historyAsync.when(
                data: (history) {
                  final activities = <String>[];
                  if (history != null) {
                    if (history.assessments.isNotEmpty) {
                      activities.add(
                          'Assessment: Match found for ${history.assessments.first.recommendedCareer}');
                    }
                    if (history.resumes.isNotEmpty) {
                      activities.add(
                          'ATS Scan: Resume Compatibility at ${history.resumes.first.atsScore}%');
                    }
                  }

                  if (activities.isEmpty) {
                    return Text(
                      'No recent activity recorded.',
                      style: TextStyle(color: Colors.grey[500], fontStyle: FontStyle.italic),
                    ).animate().fade(delay: 400.ms);
                  }

                  return Column(
                    children: activities
                        .map(
                          (act) => Container(
                            margin: const EdgeInsets.only(bottom: 16),
                            decoration: const BoxDecoration(
                              border: Border(
                                left: BorderSide(color: AppTheme.burntSienna, width: 4),
                              ),
                            ),
                            padding: const EdgeInsets.all(24),
                            color: Colors.white,
                            child: Row(
                              children: [
                                const Icon(Icons.arrow_right_alt, color: AppTheme.primaryNavy),
                                const SizedBox(width: 16),
                                Expanded(
                                  child: Text(
                                    act,
                                    style: const TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.w500,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        )
                        .toList(),
                  ).animate().fade(delay: 400.ms).slideX(begin: 0.05);
                },
                loading: () => const SizedBox.shrink(),
                error: (_, stack) => Text(
                  'Unable to load activity history.',
                  style: TextStyle(color: Colors.grey[500]),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _QuickActionTile extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  final String buttonText;
  final VoidCallback onTap;

  const _QuickActionTile({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.buttonText,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 32, color: AppTheme.primaryNavy),
          const SizedBox(height: 16),
          Text(
            title,
            style: GoogleFonts.inter(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: AppTheme.primaryNavy,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            subtitle,
            style: GoogleFonts.inter(
              fontSize: 14,
              color: Colors.grey[600],
            ),
          ),
          const SizedBox(height: 24),
          OutlinedButton(
            onPressed: onTap,
            style: OutlinedButton.styleFrom(
              foregroundColor: AppTheme.primaryNavy,
              side: const BorderSide(color: AppTheme.primaryNavy),
              shape: const RoundedRectangleBorder(borderRadius: BorderRadius.zero),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
            ),
            child: Text(
              buttonText,
              style: GoogleFonts.inter(
                fontSize: 12,
                fontWeight: FontWeight.bold,
                letterSpacing: 1,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
