import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../common/app_scaffold.dart';
import '../../providers/assessment_provider.dart';
import '../../providers/history_provider.dart';
import '../../../domain/models/assessment_model.dart';
import '../../../core/theme/app_theme.dart';

class AssessmentScreen extends ConsumerStatefulWidget {
  const AssessmentScreen({super.key});

  @override
  ConsumerState<AssessmentScreen> createState() => _AssessmentScreenState();
}

class _AssessmentScreenState extends ConsumerState<AssessmentScreen> {
  final _formKey = GlobalKey<FormState>();

  final _ageController = TextEditingController(text: '24');
  String _education = "Bachelor's";
  final _skillsController = TextEditingController(text: 'Python, SQL, Communication');
  final _interestsController = TextEditingController(text: 'Data, Analysis, Teamwork');

  double _scoreR = 5, _scoreI = 5, _scoreA = 5, _scoreS = 5, _scoreE = 5, _scoreC = 5;

  @override
  void dispose() {
    _ageController.dispose();
    _skillsController.dispose();
    _interestsController.dispose();
    super.dispose();
  }

  Future<void> _submitAssessment() async {
    if (!_formKey.currentState!.validate()) return;

    final input = AssessmentInput(
      age: int.tryParse(_ageController.text) ?? 24,
      education: _education,
      skills: _skillsController.text
          .split(',')
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList(),
      interests: _interestsController.text
          .split(',')
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList(),
      riasecScores: [_scoreR, _scoreI, _scoreA, _scoreS, _scoreE, _scoreC],
    );

    final success = await ref.read(assessmentProvider.notifier).submitAssessment(input);
    if (success) {
      ref.invalidate(historyFutureProvider);
    }
  }

  Widget _buildSlider(String label, double value, ValueChanged<double> onChanged) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
            Text(
              value.toInt().toString(),
              style: const TextStyle(
                color: AppTheme.primaryNavy,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            activeTrackColor: AppTheme.primaryNavy,
            inactiveTrackColor: Colors.grey[300],
            thumbColor: AppTheme.burntSienna,
            overlayColor: AppTheme.burntSienna.withValues(alpha: 0.2),
          ),
          child: Slider(
            value: value,
            min: 0,
            max: 10,
            divisions: 10,
            onChanged: onChanged,
          ),
        ),
        const SizedBox(height: 16),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(assessmentProvider);

    return AppScaffold(
      currentRoute: '/assessment',
      body: Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 1000),
          padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 48.0),
          child: state.result != null
              ? _buildResultView(context, state.result!)
              : Form(
                  key: _formKey,
                  child: ListView(
                    children: [
                      Text('Career Assessment Quiz.',
                              style: Theme.of(context).textTheme.displayMedium)
                          .animate()
                          .fade()
                          .slideY(begin: 0.1),
                      const SizedBox(height: 16),
                      Text(
                        'Detail your background and preferences to generate a personalized career trajectory.',
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
                            style: const TextStyle(color: Colors.red, fontWeight: FontWeight.w600),
                          ),
                        ),
                        const SizedBox(height: 24),
                      ],

                      LayoutBuilder(
                        builder: (context, constraints) {
                          final isWide = constraints.maxWidth > 700;
                          final col1 = Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('Profile Data',
                                  style: Theme.of(context).textTheme.headlineMedium),
                              const SizedBox(height: 32),
                              TextFormField(
                                controller: _ageController,
                                decoration: const InputDecoration(labelText: 'Age'),
                                keyboardType: TextInputType.number,
                              ),
                              const SizedBox(height: 24),
                              DropdownButtonFormField<String>(
                                initialValue: _education,
                                decoration: const InputDecoration(labelText: 'Highest Education'),
                                icon: const Icon(Icons.keyboard_arrow_down),
                                items: ['High School', "Bachelor's", "Master's", 'PhD', 'Self-Taught']
                                    .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                                    .toList(),
                                onChanged: (val) {
                                  if (val != null) setState(() => _education = val);
                                },
                              ),
                              const SizedBox(height: 24),
                              TextFormField(
                                controller: _skillsController,
                                decoration: const InputDecoration(
                                    labelText: 'Skills (comma separated)'),
                              ),
                              const SizedBox(height: 24),
                              TextFormField(
                                controller: _interestsController,
                                decoration: const InputDecoration(
                                    labelText: 'Interests (comma separated)'),
                              ),
                            ],
                          );

                          final col2 = Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('RIASEC Profile',
                                  style: Theme.of(context).textTheme.headlineMedium),
                              const SizedBox(height: 32),
                              _buildSlider(
                                  'Realistic', _scoreR, (v) => setState(() => _scoreR = v)),
                              _buildSlider(
                                  'Investigative', _scoreI, (v) => setState(() => _scoreI = v)),
                              _buildSlider(
                                  'Artistic', _scoreA, (v) => setState(() => _scoreA = v)),
                              _buildSlider(
                                  'Social', _scoreS, (v) => setState(() => _scoreS = v)),
                              _buildSlider(
                                  'Enterprising', _scoreE, (v) => setState(() => _scoreE = v)),
                              _buildSlider(
                                  'Conventional', _scoreC, (v) => setState(() => _scoreC = v)),
                            ],
                          );

                          if (isWide) {
                            return Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Expanded(child: col1),
                                const SizedBox(width: 64),
                                Expanded(child: col2),
                              ],
                            );
                          } else {
                            return Column(
                              children: [col1, const SizedBox(height: 48), col2],
                            );
                          }
                        },
                      ).animate().fade(delay: 200.ms).slideY(begin: 0.1),

                      const SizedBox(height: 64),
                      Center(
                        child: SizedBox(
                          width: 300,
                          child: FilledButton(
                            onPressed: state.isLoading ? null : _submitAssessment,
                            child: state.isLoading
                                ? const SizedBox(
                                    width: 20,
                                    height: 20,
                                    child: CircularProgressIndicator(
                                      color: Colors.white,
                                      strokeWidth: 2,
                                    ),
                                  )
                                : const Text('ANALYZE PROFILE'),
                          ),
                        ),
                      ).animate().fade(delay: 300.ms),
                      const SizedBox(height: 64),
                    ],
                  ),
                ),
        ),
      ),
    );
  }

  Widget _buildResultView(BuildContext context, AssessmentResult result) {
    return ListView(
      children: [
        Text('Analysis Complete.', style: Theme.of(context).textTheme.displayMedium)
            .animate()
            .fade()
            .slideY(begin: 0.1),
        const SizedBox(height: 48),
        Container(
          padding: const EdgeInsets.all(48),
          color: Colors.white,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'PRIMARY MATCH',
                style: GoogleFonts.inter(
                  color: AppTheme.burntSienna,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 2,
                  fontSize: 12,
                ),
              ),
              const SizedBox(height: 16),
              Text(
                result.recommendedCareer,
                style: GoogleFonts.playfairDisplay(
                  fontSize: 48,
                  fontWeight: FontWeight.w700,
                  color: AppTheme.primaryNavy,
                  height: 1.1,
                ),
              ),
              const SizedBox(height: 24),
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    color: AppTheme.primaryNavy,
                    child: Text(
                      'Confidence: ${(result.recommendationScore * 100).toStringAsFixed(1)}%',
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.grey[300]!),
                    ),
                    child: Text(
                      'Cluster #${result.cluster}',
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 48),
              const Divider(),
              const SizedBox(height: 48),
              Text(
                'STRONG ALTERNATIVES',
                style: GoogleFonts.inter(
                  color: Colors.grey[500],
                  fontWeight: FontWeight.w700,
                  letterSpacing: 2,
                  fontSize: 12,
                ),
              ),
              const SizedBox(height: 24),
              for (var alt in result.topCareers)
                Padding(
                  padding: const EdgeInsets.only(bottom: 16.0),
                  child: Row(
                    children: [
                      const Icon(Icons.arrow_right_alt, color: AppTheme.burntSienna),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Text(
                          alt.career,
                          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
                        ),
                      ),
                      Text(
                        '${(alt.score * 100).toStringAsFixed(1)}%',
                        style: TextStyle(color: Colors.grey[500], fontSize: 18),
                      ),
                    ],
                  ),
                ),
            ],
          ),
        ).animate().fade(delay: 200.ms).slideY(begin: 0.1),
        const SizedBox(height: 48),
        Center(
          child: TextButton.icon(
            onPressed: () => ref.read(assessmentProvider.notifier).reset(),
            icon: const Icon(Icons.refresh, color: AppTheme.primaryNavy),
            label: const Text(
              'NEW ASSESSMENT',
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
