import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:url_launcher/url_launcher.dart';
import '../../common/app_scaffold.dart';
import '../../providers/resume_provider.dart';
import '../../providers/history_provider.dart';
import '../../../domain/models/resume_model.dart';
import '../../../core/theme/app_theme.dart';

class ResumeScreen extends ConsumerStatefulWidget {
  const ResumeScreen({super.key});

  @override
  ConsumerState<ResumeScreen> createState() => _ResumeScreenState();
}

class _ResumeScreenState extends ConsumerState<ResumeScreen> {
  final _formKey = GlobalKey<FormState>();
  final _resumeController = TextEditingController();
  final _targetCareerController = TextEditingController(text: 'Data Scientist');

  @override
  void dispose() {
    _resumeController.dispose();
    _targetCareerController.dispose();
    super.dispose();
  }

  Future<void> _analyzeResume() async {
    if (!_formKey.currentState!.validate()) return;

    final success = await ref.read(resumeProvider.notifier).analyzeResume(
          resumeText: _resumeController.text,
          targetCareer: _targetCareerController.text,
        );

    if (success) {
      ref.invalidate(historyFutureProvider);
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(resumeProvider);

    return AppScaffold(
      currentRoute: '/resume',
      body: Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 800),
          padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 48.0),
          child: state.result != null
              ? _buildResultView(context, state.result!)
              : Form(
                  key: _formKey,
                  child: ListView(
                    children: [
                      Text('ATS Resume Scan.',
                              style: Theme.of(context).textTheme.displayMedium)
                          .animate()
                          .fade()
                          .slideY(begin: 0.1),
                      const SizedBox(height: 16),
                      Text(
                        'Upload your resume text to evaluate compatibility against your target role.',
                        style: Theme.of(context)
                            .textTheme
                            .bodyLarge
                            ?.copyWith(fontSize: 18),
                      ).animate().fade(delay: 100.ms).slideY(begin: 0.1),
                      const SizedBox(height: 48),

                      if (state.errorMessage != null) ...[
                        Container(
                          padding: const EdgeInsets.all(16),
                          color: Colors.red.withValues(alpha: 0.1),
                          child: Text(
                            state.errorMessage!,
                            style: const TextStyle(
                                color: Colors.red, fontWeight: FontWeight.w600),
                          ),
                        ),
                        const SizedBox(height: 24),
                      ],

                      TextFormField(
                        controller: _targetCareerController,
                        decoration: const InputDecoration(
                          labelText: 'Target Role (e.g. Data Scientist, UX Designer)',
                        ),
                        validator: (val) =>
                            val == null || val.isEmpty ? 'Required' : null,
                      ).animate().fade(delay: 200.ms).slideY(begin: 0.1),
                      const SizedBox(height: 32),

                      TextFormField(
                        controller: _resumeController,
                        decoration: const InputDecoration(
                          labelText: 'Paste Resume Content Here',
                          alignLabelWithHint: true,
                        ),
                        maxLines: 12,
                        validator: (val) =>
                            val == null || val.isEmpty ? 'Required' : null,
                      ).animate().fade(delay: 300.ms).slideY(begin: 0.1),
                      const SizedBox(height: 48),

                      Center(
                        child: SizedBox(
                          width: 300,
                          child: FilledButton(
                            onPressed: state.isLoading ? null : _analyzeResume,
                            child: state.isLoading
                                ? const SizedBox(
                                    width: 20,
                                    height: 20,
                                    child: CircularProgressIndicator(
                                      color: Colors.white,
                                      strokeWidth: 2,
                                    ),
                                  )
                                : const Text('INITIATE SCAN'),
                          ),
                        ),
                      ).animate().fade(delay: 400.ms),
                    ],
                  ),
                ),
        ),
      ),
    );
  }

  Widget _buildResultView(BuildContext context, ResumeAnalysisResult analysis) {
    final atsScore = analysis.atsScore;

    return ListView(
      children: [
        Text('Scan Complete.', style: Theme.of(context).textTheme.displayMedium)
            .animate()
            .fade()
            .slideY(begin: 0.1),
        const SizedBox(height: 48),

        Container(
          color: Colors.white,
          padding: const EdgeInsets.all(48),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'OVERALL MATCH',
                          style: GoogleFonts.inter(
                            color: Colors.grey[500],
                            fontWeight: FontWeight.w700,
                            letterSpacing: 2,
                            fontSize: 12,
                          ),
                        ),
                        const SizedBox(height: 16),
                        Text(
                          analysis.recommendation,
                          style: Theme.of(context)
                              .textTheme
                              .headlineMedium
                              ?.copyWith(height: 1.3),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 48),
                  Container(
                    padding: const EdgeInsets.all(32),
                    decoration: BoxDecoration(
                      border: Border.all(
                        color: atsScore > 75
                            ? AppTheme.softBlue
                            : (atsScore > 50 ? AppTheme.burntSienna : Colors.red),
                        width: 4,
                      ),
                      shape: BoxShape.circle,
                    ),
                    child: Text(
                      '${atsScore.toInt()}%',
                      style: GoogleFonts.playfairDisplay(
                        fontSize: 48,
                        fontWeight: FontWeight.bold,
                        color: AppTheme.primaryNavy,
                      ),
                    ),
                  )
                ],
              ),
              const SizedBox(height: 64),

              LayoutBuilder(builder: (context, constraints) {
                final isWide = constraints.maxWidth > 500;
                final verified = Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'VERIFIED SKILLS',
                      style: GoogleFonts.inter(
                        color: AppTheme.softBlue,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 2,
                        fontSize: 12,
                      ),
                    ),
                    const SizedBox(height: 24),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: analysis.skillsFound
                          .map((s) => Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 16, vertical: 8),
                                color: AppTheme.softBlue.withValues(alpha: 0.1),
                                child: Text(
                                  s,
                                  style: const TextStyle(
                                    color: AppTheme.primaryNavy,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ))
                          .toList(),
                    )
                  ],
                );

                final missing = Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'MISSING SKILLS',
                      style: GoogleFonts.inter(
                        color: AppTheme.burntSienna,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 2,
                        fontSize: 12,
                      ),
                    ),
                    const SizedBox(height: 24),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: analysis.skillsMissing
                          .map((s) => Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 16, vertical: 8),
                                color: AppTheme.burntSienna.withValues(alpha: 0.1),
                                child: Text(
                                  s,
                                  style: const TextStyle(
                                    color: AppTheme.burntSienna,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ))
                          .toList(),
                    ),
                  ],
                );

                if (isWide) {
                  return Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(child: verified),
                      const SizedBox(width: 48),
                      Expanded(child: missing),
                    ],
                  );
                } else {
                  return Column(
                    children: [verified, const SizedBox(height: 48), missing],
                  );
                }
              }),

              if (analysis.interviewQuestions.isNotEmpty) ...[
                const SizedBox(height: 64),
                const Divider(),
                const SizedBox(height: 64),
                Text(
                  'INTERVIEW PREPARATION',
                  style: GoogleFonts.inter(
                    color: Colors.grey[500],
                    fontWeight: FontWeight.w700,
                    letterSpacing: 2,
                    fontSize: 12,
                  ),
                ),
                const SizedBox(height: 32),
                Text('Tailored Questions', style: Theme.of(context).textTheme.titleLarge),
                const SizedBox(height: 16),
                for (var q in analysis.interviewQuestions)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 12.0),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Padding(
                          padding: EdgeInsets.only(top: 8),
                          child: Icon(Icons.circle, size: 8, color: AppTheme.primaryNavy),
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: Text(
                            q,
                            style: const TextStyle(fontSize: 16, height: 1.5),
                          ),
                        ),
                      ],
                    ),
                  ),

                if (analysis.roadmapUrl != null) ...[
                  const SizedBox(height: 48),
                  Center(
                    child: OutlinedButton.icon(
                      onPressed: () async {
                        final url = Uri.parse(analysis.roadmapUrl!);
                        if (await canLaunchUrl(url)) await launchUrl(url);
                      },
                      icon: const Icon(Icons.map, color: AppTheme.primaryNavy),
                      label: const Text(
                        'VIEW ROADMAP',
                        style: TextStyle(
                          color: AppTheme.primaryNavy,
                          letterSpacing: 1,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 32, vertical: 24),
                        side: const BorderSide(color: AppTheme.primaryNavy, width: 2),
                        shape: const RoundedRectangleBorder(
                            borderRadius: BorderRadius.zero),
                      ),
                    ),
                  ),
                ]
              ]
            ],
          ),
        ).animate().fade(delay: 200.ms).slideY(begin: 0.1),

        const SizedBox(height: 48),
        Center(
          child: TextButton.icon(
            onPressed: () => ref.read(resumeProvider.notifier).reset(),
            icon: const Icon(Icons.refresh, color: AppTheme.primaryNavy),
            label: const Text(
              'NEW SCAN',
              style: TextStyle(
                color: AppTheme.primaryNavy,
                letterSpacing: 1,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ).animate().fade(delay: 300.ms),
      ],
    );
  }
}
