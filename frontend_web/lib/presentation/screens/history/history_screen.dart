import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../common/app_scaffold.dart';
import '../../providers/history_provider.dart';
import '../../../core/theme/app_theme.dart';

class HistoryScreen extends ConsumerWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final historyAsync = ref.watch(historyFutureProvider);

    return AppScaffold(
      currentRoute: '/history',
      body: Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 800),
          padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 48.0),
          child: historyAsync.when(
            data: (history) {
              final assessments = history?.assessments ?? [];
              final resumes = history?.resumes ?? [];

              return ListView(
                children: [
                  Text('History Timeline.',
                          style: Theme.of(context).textTheme.displayMedium)
                      .animate()
                      .fade()
                      .slideY(begin: 0.1),
                  const SizedBox(height: 16),
                  Text(
                    'A complete timeline of your career recommendations and ATS scans.',
                    style: Theme.of(context)
                        .textTheme
                        .bodyLarge
                        ?.copyWith(fontSize: 18),
                  ).animate().fade(delay: 100.ms).slideY(begin: 0.1),
                  const SizedBox(height: 48),

                  Text(
                    'ASSESSMENTS',
                    style: GoogleFonts.inter(
                      color: Colors.grey[500],
                      fontWeight: FontWeight.w700,
                      letterSpacing: 2,
                      fontSize: 12,
                    ),
                  ).animate().fade(delay: 200.ms),
                  const SizedBox(height: 24),
                  if (assessments.isEmpty)
                    Text(
                      'No assessments on record.',
                      style: TextStyle(color: Colors.grey[500], fontStyle: FontStyle.italic),
                    ).animate().fade(delay: 200.ms)
                  else
                    for (var a in assessments)
                      Container(
                        margin: const EdgeInsets.only(bottom: 16),
                        padding: const EdgeInsets.all(32),
                        color: Colors.white,
                        child: Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.all(16),
                              color: AppTheme.backgroundColor,
                              child: const Icon(Icons.psychology,
                                  color: AppTheme.primaryNavy),
                            ),
                            const SizedBox(width: 24),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    a.recommendedCareer,
                                    style: const TextStyle(
                                      fontSize: 20,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    'Confidence: ${(a.recommendationScore * 100).toStringAsFixed(1)}%',
                                    style: TextStyle(color: Colors.grey[600]),
                                  ),
                                ],
                              ),
                            ),
                            Text(
                              a.createdAt.contains('T')
                                  ? a.createdAt.split('T')[0]
                                  : a.createdAt,
                              style: TextStyle(
                                color: Colors.grey[400],
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ).animate().fade(delay: 250.ms).slideX(begin: 0.05),

                  const SizedBox(height: 64),
                  Text(
                    'ATS SCANS',
                    style: GoogleFonts.inter(
                      color: Colors.grey[500],
                      fontWeight: FontWeight.w700,
                      letterSpacing: 2,
                      fontSize: 12,
                    ),
                  ).animate().fade(delay: 300.ms),
                  const SizedBox(height: 24),
                  if (resumes.isEmpty)
                    Text(
                      'No resumes on record.',
                      style: TextStyle(color: Colors.grey[500], fontStyle: FontStyle.italic),
                    ).animate().fade(delay: 300.ms)
                  else
                    for (var r in resumes)
                      Container(
                        margin: const EdgeInsets.only(bottom: 16),
                        padding: const EdgeInsets.all(32),
                        color: Colors.white,
                        child: Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.all(16),
                              color: AppTheme.backgroundColor,
                              child: const Icon(Icons.document_scanner,
                                  color: AppTheme.primaryNavy),
                            ),
                            const SizedBox(width: 24),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Text(
                                    'ATS Compatibility',
                                    style: TextStyle(
                                      fontSize: 20,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    'Score: ${r.atsScore}%',
                                    style: TextStyle(color: Colors.grey[600]),
                                  ),
                                ],
                              ),
                            ),
                            Text(
                              r.createdAt.contains('T')
                                  ? r.createdAt.split('T')[0]
                                  : r.createdAt,
                              style: TextStyle(
                                color: Colors.grey[400],
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ).animate().fade(delay: 350.ms).slideX(begin: 0.05),
                ],
              );
            },
            loading: () => const Center(
              child: CircularProgressIndicator(color: AppTheme.primaryNavy),
            ),
            error: (err, stack) => Center(
              child: Text('Failed to load history: $err'),
            ),
          ),
        ),
      ),
    );
  }
}
