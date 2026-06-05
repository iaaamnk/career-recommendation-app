import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter/foundation.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';

void main() {
  runApp(const ProviderScope(child: PathFinderApp()));
}

// ---------------- STATE PROVIDERS ----------------
class DashboardData {
  final int assessmentsTaken;
  final String topRecommendation;
  final int atsScore;
  final List<String> recentActivity;
  DashboardData({this.assessmentsTaken=0, this.topRecommendation='None', this.atsScore=0, this.recentActivity=const []});
}

class DashboardNotifier extends Notifier<DashboardData> {
  @override
  DashboardData build() => DashboardData();

  void updateAssessment(String recommendation) {
    state = DashboardData(
      assessmentsTaken: state.assessmentsTaken + 1,
      topRecommendation: recommendation,
      atsScore: state.atsScore,
      recentActivity: ['Completed Holland RIASEC Assessment', ...state.recentActivity],
    );
  }

  void updateAtsScore(int score) {
    state = DashboardData(
      assessmentsTaken: state.assessmentsTaken,
      topRecommendation: state.topRecommendation,
      atsScore: score,
      recentActivity: ['Completed ATS Resume Check: $score%', ...state.recentActivity],
    );
  }
}

final dashboardProvider = NotifierProvider<DashboardNotifier, DashboardData>(DashboardNotifier.new);



// ---------------- MAIN APP WIDGET ----------------
class PathFinderApp extends StatelessWidget {
  const PathFinderApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PathFinder AI',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6C63FF), // A nice vibrant purple/blue
          brightness: Brightness.light,
        ),
        textTheme: const TextTheme(
          displayLarge: TextStyle(fontWeight: FontWeight.bold, letterSpacing: -1.0),
          titleLarge: TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
      home: const MainLayout(),
    );
  }
}

// ---------------- LAYOUT WIDGET ----------------
class MainLayout extends StatefulWidget {
  const MainLayout({super.key});

  @override
  State<MainLayout> createState() => _MainLayoutState();
}

class _MainLayoutState extends State<MainLayout> {
  String currentView = 'dashboard';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.explore, color: Color(0xFF6C63FF)),
            SizedBox(width: 8),
            Text('PathFinder', style: TextStyle(fontWeight: FontWeight.w900)),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => setState(() => currentView = 'dashboard'),
            child: const Text('Dashboard'),
          ),
          TextButton(
            onPressed: () => setState(() => currentView = 'assessment'),
            child: const Text('Assessment'),
          ),
          TextButton(
            onPressed: () => setState(() => currentView = 'resume'),
            child: const Text('ATS Check'),
          ),
          TextButton(
            onPressed: () => setState(() => currentView = 'profile'),
            child: const Text('Profile'),
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: currentView == 'dashboard' 
          ? const DashboardView() 
          : currentView == 'assessment' 
              ? const AssessmentView() 
              : currentView == 'resume'
                  ? const ResumeAnalysisView()
                  : const ProfileView(),
    );
  }
}

// ---------------- DASHBOARD VIEW ----------------
class DashboardView extends ConsumerWidget {
  const DashboardView({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final dashboardData = ref.watch(dashboardProvider);
    return Center(
      child: Container(
        constraints: const BoxConstraints(maxWidth: 900),
        padding: const EdgeInsets.all(24.0),
        child: ListView(
          children: [
            Text('Welcome back, User', style: Theme.of(context).textTheme.headlineMedium),
            const SizedBox(height: 8),
            Text('Here is your AI-powered career overview.', style: Theme.of(context).textTheme.bodyLarge?.copyWith(color: Colors.grey[700])),
            const SizedBox(height: 32),
            Row(
              children: [
                Expanded(child: _DashboardCard(title: 'Assessments Taken', value: dashboardData.assessmentsTaken.toString(), icon: Icons.assignment_turned_in)),
                const SizedBox(width: 16),
                Expanded(child: _DashboardCard(title: 'Top Recommendation', value: dashboardData.topRecommendation, icon: Icons.star)),
                const SizedBox(width: 16),
                Expanded(child: _DashboardCard(title: 'ATS Resume Score', value: dashboardData.atsScore > 0 ? '${dashboardData.atsScore}%' : 'N/A', icon: Icons.document_scanner)),
              ],
            ),
            const SizedBox(height: 32),
            Text('Recent Activity', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 16),
            if (dashboardData.recentActivity.isEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 32.0),
                child: Center(child: Text("No recent activity.", style: TextStyle(color: Colors.grey[600]))),
              )
            else
              ListView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: dashboardData.recentActivity.length,
                itemBuilder: (context, index) {
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 8.0),
                    child: ListTile(
                      leading: const CircleAvatar(child: Icon(Icons.analytics)),
                      title: Text(dashboardData.recentActivity[index]),
                      subtitle: const Text('Just now'),
                      trailing: const Icon(Icons.chevron_right),
                      tileColor: Theme.of(context).colorScheme.surfaceContainerHighest,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                  );
                },
              ),
          ],
        ),
      ),
    );
  }
}

class _DashboardCard extends StatelessWidget {
  final String title;
  final String value;
  final IconData icon;

  const _DashboardCard({required this.title, required this.value, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      color: Theme.of(context).colorScheme.primaryContainer,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: Theme.of(context).colorScheme.primary),
            const SizedBox(height: 16),
            Text(title, style: Theme.of(context).textTheme.bodyMedium),
            const SizedBox(height: 4),
            Text(value, style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
          ],
        ),
      ),
    );
  }
}

// ---------------- ASSESSMENT VIEW ----------------
class AssessmentView extends ConsumerStatefulWidget {
  const AssessmentView({super.key});

  @override
  ConsumerState<AssessmentView> createState() => _AssessmentViewState();
}

class _AssessmentViewState extends ConsumerState<AssessmentView> {
  final _formKey = GlobalKey<FormState>();
  bool isLoading = false;
  Map<String, dynamic>? result;

  // Form Fields
  final _ageController = TextEditingController(text: '24');
  String _education = "Bachelor's";
  final _skillsController = TextEditingController(text: 'Python, SQL');
  final _interestsController = TextEditingController(text: 'Data, Analysis');

  // RIASEC Scores (0-10)
  double _scoreR = 5;
  double _scoreI = 5;
  double _scoreA = 5;
  double _scoreS = 5;
  double _scoreE = 5;
  double _scoreC = 5;

  Future<void> _submitAssessment() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => isLoading = true);
    try {
      final response = await http.post(
        Uri.parse('https://career-recommendation-app-2-08ny.onrender.com/api/recommend'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          "user_id": 1, // Mock user ID
          "age": int.tryParse(_ageController.text) ?? 24,
          "education": _education,
          "skills": _skillsController.text.split(',').map((e) => e.trim()).where((e) => e.isNotEmpty).toList(),
          "interests": _interestsController.text.split(',').map((e) => e.trim()).where((e) => e.isNotEmpty).toList(),
          "riasec_scores": [_scoreR, _scoreI, _scoreA, _scoreS, _scoreE, _scoreC]
        }),
      );

      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        setState(() => result = decoded);
        // Update Riverpod state
        ref.read(dashboardProvider.notifier).updateAssessment(decoded['prediction']['Recommended_Career']);
      } else {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: ${response.statusCode}')));
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to connect: $e')));
    } finally {
      setState(() => isLoading = false);
    }
  }

  Widget _buildSlider(String label, double value, ValueChanged<double> onChanged) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('$label (${value.toInt()})', style: const TextStyle(fontWeight: FontWeight.w600)),
        Slider(
          value: value,
          min: 0,
          max: 10,
          divisions: 10,
          label: value.toInt().toString(),
          onChanged: onChanged,
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    if (result != null) {
      return Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 600),
          padding: const EdgeInsets.all(24.0),
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _ResultCard(result: result!),
              const SizedBox(height: 24),
              TextButton.icon(
                onPressed: () => setState(() => result = null),
                icon: const Icon(Icons.refresh),
                label: const Text('Take Another Assessment'),
              )
              ],
            ),
          ),
        ),
      );
    }

    return Center(
      child: Container(
        constraints: const BoxConstraints(maxWidth: 700),
        padding: const EdgeInsets.all(24.0),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              const Icon(Icons.psychology, size: 48, color: Color(0xFF6C63FF)),
              const SizedBox(height: 16),
              Text('Career Assessment Profile', style: Theme.of(context).textTheme.headlineMedium, textAlign: TextAlign.center),
              const SizedBox(height: 8),
              const Text('Fill out your details to generate your personalized AI career roadmap.', textAlign: TextAlign.center),
              const SizedBox(height: 32),
              
              // Basic Info
              Card(
                elevation: 0,
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Basic Information', style: Theme.of(context).textTheme.titleLarge),
                      const SizedBox(height: 16),
                      TextFormField(
                        controller: _ageController,
                        decoration: const InputDecoration(labelText: 'Age', border: OutlineInputBorder()),
                        keyboardType: TextInputType.number,
                        validator: (value) => value!.isEmpty ? 'Required' : null,
                      ),
                      const SizedBox(height: 16),
                      DropdownButtonFormField<String>(
                        value: _education,
                        decoration: const InputDecoration(labelText: 'Education Level', border: OutlineInputBorder()),
                        items: ['High School', "Bachelor's", "Master's", 'PhD', 'Self-Taught']
                            .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                            .toList(),
                        onChanged: (val) => setState(() => _education = val!),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // Skills & Interests
              Card(
                elevation: 0,
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Skills & Interests', style: Theme.of(context).textTheme.titleLarge),
                      const SizedBox(height: 16),
                      TextFormField(
                        controller: _skillsController,
                        decoration: const InputDecoration(labelText: 'Skills (comma separated)', hintText: 'e.g. Python, Marketing, Design', border: OutlineInputBorder()),
                      ),
                      const SizedBox(height: 16),
                      TextFormField(
                        controller: _interestsController,
                        decoration: const InputDecoration(labelText: 'Interests (comma separated)', hintText: 'e.g. Data Analysis, Writing', border: OutlineInputBorder()),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // RIASEC Scores
              Card(
                elevation: 0,
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Holland RIASEC Scores (0-10)', style: Theme.of(context).textTheme.titleLarge),
                      const SizedBox(height: 16),
                      _buildSlider('Realistic (Doer)', _scoreR, (v) => setState(() => _scoreR = v)),
                      _buildSlider('Investigative (Thinker)', _scoreI, (v) => setState(() => _scoreI = v)),
                      _buildSlider('Artistic (Creator)', _scoreA, (v) => setState(() => _scoreA = v)),
                      _buildSlider('Social (Helper)', _scoreS, (v) => setState(() => _scoreS = v)),
                      _buildSlider('Enterprising (Persuader)', _scoreE, (v) => setState(() => _scoreE = v)),
                      _buildSlider('Conventional (Organizer)', _scoreC, (v) => setState(() => _scoreC = v)),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 32),
              
              FilledButton.icon(
                onPressed: isLoading ? null : _submitAssessment,
                icon: isLoading ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)) : const Icon(Icons.auto_awesome),
                label: const Text('Analyze My Profile'),
                style: FilledButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 20)),
              ),
              const SizedBox(height: 48),
            ],
          ),
        ),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  final Map<String, dynamic> result;

  const _ResultCard({required this.result});

  @override
  Widget build(BuildContext context) {
    final prediction = result['prediction'];
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                const Icon(Icons.check_circle, color: Colors.green, size: 32),
                const SizedBox(width: 12),
                Text('Analysis Complete', style: Theme.of(context).textTheme.headlineSmall),
              ],
            ),
            const Divider(height: 48),
            Text('Top Recommendation (Supervised AI)', style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey[700])),
            const SizedBox(height: 8),
            Text(prediction['Recommended_Career'], style: Theme.of(context).textTheme.displaySmall?.copyWith(fontWeight: FontWeight.bold, color: Theme.of(context).colorScheme.primary)),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.green.withOpacity(0.1),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text('Confidence: ${(prediction['Recommendation_Score'] * 100).toStringAsFixed(1)}%', style: const TextStyle(color: Colors.green, fontWeight: FontWeight.bold)),
            ),
            
            const SizedBox(height: 32),
            Text('Cluster Profile (Unsupervised AI)', style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey[700])),
            const SizedBox(height: 8),
            Text(prediction['Unsupervised_Recommendation'], style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold, color: Colors.orange)),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.orange.withOpacity(0.1),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text('Cluster #${prediction['Unsupervised_Cluster']}', style: const TextStyle(color: Colors.orange, fontWeight: FontWeight.bold)),
            ),

            const SizedBox(height: 32),
            Text('Strong Alternatives', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            for (var alt in prediction['Top_3_Careers'])
              Padding(
                padding: const EdgeInsets.only(bottom: 8.0),
                child: Row(
                  children: [
                    const Icon(Icons.arrow_right, color: Colors.grey),
                    Text('${alt['career']}', style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500)),
                    const Spacer(),
                    Text('${(alt['score'] * 100).toStringAsFixed(1)}%', style: TextStyle(color: Colors.grey[600])),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// ---------------- RESUME ANALYSIS VIEW ----------------
class ResumeAnalysisView extends ConsumerStatefulWidget {
  const ResumeAnalysisView({super.key});

  @override
  ConsumerState<ResumeAnalysisView> createState() => _ResumeAnalysisViewState();
}

class _ResumeAnalysisViewState extends ConsumerState<ResumeAnalysisView> {
  final _formKey = GlobalKey<FormState>();
  bool isLoading = false;
  Map<String, dynamic>? result;

  final _resumeController = TextEditingController();
  final _targetCareerController = TextEditingController(text: 'Data Scientist');

  Future<void> _analyzeResume() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => isLoading = true);
    try {
      final response = await http.post(
        Uri.parse('https://career-recommendation-app-2-08ny.onrender.com/api/resume/analyze'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          "user_id": 1, // Mock user ID
          "resume_text": _resumeController.text,
          "target_career": _targetCareerController.text,
        }),
      );

      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        setState(() => result = decoded);
        // Update Riverpod state
        ref.read(dashboardProvider.notifier).updateAtsScore((decoded['analysis']['ats_score'] as double).toInt());
      } else {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: ${response.statusCode}')));
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to connect: $e')));
    } finally {
      setState(() => isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (result != null) {
      final analysis = result!['analysis'];
      final atsScore = analysis['ats_score'] as double;
      
      return Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 800),
          padding: const EdgeInsets.all(24.0),
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Card(
                  elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                child: Padding(
                  padding: const EdgeInsets.all(32.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(Icons.document_scanner, size: 32, color: Theme.of(context).colorScheme.primary),
                          const SizedBox(width: 12),
                          Text('ATS Analysis Report', style: Theme.of(context).textTheme.headlineSmall),
                        ],
                      ),
                      const Divider(height: 48),
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.center,
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text('Overall ATS Match', style: Theme.of(context).textTheme.titleLarge),
                                const SizedBox(height: 8),
                                Text(analysis['recommendation'], style: TextStyle(color: Colors.grey[700])),
                              ],
                            ),
                          ),
                          Container(
                            width: 100,
                            height: 100,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              border: Border.all(
                                color: atsScore > 75 ? Colors.green : (atsScore > 50 ? Colors.orange : Colors.red),
                                width: 8,
                              ),
                            ),
                            alignment: Alignment.center,
                            child: Text('${atsScore.toInt()}%', style: Theme.of(context).textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.bold)),
                          ),
                        ],
                      ),
                      const SizedBox(height: 32),
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text('Skills Matched', style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.green)),
                                const SizedBox(height: 12),
                                Wrap(
                                  spacing: 8,
                                  runSpacing: 8,
                                  children: (analysis['skills_found'] as List).map<Widget>((s) => Chip(
                                    label: Text(s),
                                    backgroundColor: Colors.green.withOpacity(0.1),
                                    side: const BorderSide(color: Colors.transparent),
                                  )).toList(),
                                )
                              ],
                            ),
                          ),
                          const SizedBox(width: 24),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text('Skill Gaps Identified', style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.red)),
                                const SizedBox(height: 12),
                                Wrap(
                                  spacing: 8,
                                  runSpacing: 8,
                                  children: (analysis['skills_missing'] as List).map<Widget>((s) => Chip(
                                    label: Text(s),
                                    backgroundColor: Colors.red.withOpacity(0.1),
                                    side: const BorderSide(color: Colors.transparent),
                                  )).toList(),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                        
                        // New: Overall AI Analysis Section
                        if (analysis['overall_analysis'] != null) ...[
                          const SizedBox(height: 32),
                          Container(
                            padding: const EdgeInsets.all(24),
                            decoration: BoxDecoration(
                              color: Theme.of(context).colorScheme.secondaryContainer.withOpacity(0.5),
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(color: Theme.of(context).colorScheme.secondary.withOpacity(0.3)),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Icon(Icons.auto_awesome, color: Theme.of(context).colorScheme.secondary),
                                    const SizedBox(width: 8),
                                    Text('AI Overall Analysis', style: Theme.of(context).textTheme.titleLarge?.copyWith(color: Theme.of(context).colorScheme.secondary)),
                                  ],
                                ),
                                const SizedBox(height: 12),
                                Text(analysis['overall_analysis'], style: const TextStyle(fontSize: 16, height: 1.5)),
                              ],
                            ),
                          ),
                        ],

                        // New: Interview Prep & Roadmap
                        if (result!['interview_prep'] != null) ...[
                          const SizedBox(height: 32),
                          const Divider(),
                          const SizedBox(height: 16),
                          Row(
                            children: [
                              Icon(Icons.psychology, size: 28, color: Theme.of(context).colorScheme.primary),
                              const SizedBox(width: 8),
                              Text('Interview Preparation', style: Theme.of(context).textTheme.headlineSmall),
                            ],
                          ),
                          const SizedBox(height: 16),
                          Text('Tailored Questions:', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
                          const SizedBox(height: 8),
                          ...(result!['interview_prep']['interview_questions'] as List).map((q) => Padding(
                            padding: const EdgeInsets.only(bottom: 6.0, left: 8.0),
                            child: Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text('• ', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                                Expanded(child: Text(q, style: const TextStyle(fontSize: 15))),
                              ],
                            ),
                          )).toList(),
                          
                          const SizedBox(height: 16),
                          Text('Pro Tips:', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
                          const SizedBox(height: 8),
                          ...(result!['interview_prep']['tips'] as List).map((t) => Padding(
                            padding: const EdgeInsets.only(bottom: 6.0, left: 8.0),
                            child: Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Icon(Icons.lightbulb_outline, size: 16, color: Colors.amber),
                                const SizedBox(width: 8),
                                Expanded(child: Text(t, style: const TextStyle(fontSize: 15, fontStyle: FontStyle.italic))),
                              ],
                            ),
                          )).toList(),
                          
                          const SizedBox(height: 32),
                          Center(
                            child: FilledButton.tonalIcon(
                              onPressed: () async {
                                final url = Uri.parse(result!['interview_prep']['roadmap_url']);
                                if (await canLaunchUrl(url)) {
                                  await launchUrl(url);
                                } else {
                                  if (context.mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Could not launch roadmap URL')));
                                }
                              },
                              icon: const Icon(Icons.map),
                              label: const Text('View Career Roadmap on roadmap.sh'),
                              style: FilledButton.styleFrom(
                                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                                textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)
                              ),
                            ),
                          ),
                        ]
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 24),
                TextButton.icon(
                  onPressed: () => setState(() => result = null),
                  icon: const Icon(Icons.refresh),
                  label: const Text('Analyze Another Resume'),
                )
              ],
            ),
          ),
        ),
      );
    }

    return Center(
      child: Container(
        constraints: const BoxConstraints(maxWidth: 700),
        padding: const EdgeInsets.all(24.0),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              const Icon(Icons.document_scanner, size: 48, color: Color(0xFF6C63FF)),
              const SizedBox(height: 16),
              Text('AI Resume Analysis', style: Theme.of(context).textTheme.headlineMedium, textAlign: TextAlign.center),
              const SizedBox(height: 8),
              const Text('Paste your resume text below to generate an ATS score and skill gap analysis against your target role.', textAlign: TextAlign.center),
              const SizedBox(height: 32),
              
              TextFormField(
                controller: _targetCareerController,
                decoration: const InputDecoration(labelText: 'Target Career Role', border: OutlineInputBorder(), prefixIcon: Icon(Icons.work)),
                validator: (value) => value!.isEmpty ? 'Required' : null,
              ),
              const SizedBox(height: 24),
              TextFormField(
                controller: _resumeController,
                decoration: const InputDecoration(
                  labelText: 'Paste Resume Text Here',
                  border: OutlineInputBorder(),
                  alignLabelWithHint: true,
                ),
                maxLines: 12,
                validator: (value) => value!.isEmpty ? 'Required' : null,
              ),
              const SizedBox(height: 32),
              
              FilledButton.icon(
                onPressed: isLoading ? null : _analyzeResume,
                icon: isLoading ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)) : const Icon(Icons.analytics),
                label: const Text('Scan & Score Resume'),
                style: FilledButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 20)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------- PROFILE VIEW ----------------
class ProfileView extends StatefulWidget {
  const ProfileView({super.key});

  @override
  State<ProfileView> createState() => _ProfileViewState();
}

class _ProfileViewState extends State<ProfileView> {
  final _nameController = TextEditingController(text: 'Demo User');
  final _emailController = TextEditingController(text: 'demo@example.com');
  bool _isSaved = false;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Container(
        constraints: const BoxConstraints(maxWidth: 800),
        padding: const EdgeInsets.all(24.0),
        child: ListView(
          children: [
            const Icon(Icons.person, size: 48, color: Color(0xFF6C63FF)),
            const SizedBox(height: 16),
            Text('Profile Creation', style: Theme.of(context).textTheme.headlineMedium, textAlign: TextAlign.center),
            const SizedBox(height: 32),

            // Profile Form
            Card(
              elevation: 0,
              color: Theme.of(context).colorScheme.surfaceContainerHighest,
              child: Padding(
                padding: const EdgeInsets.all(24.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Your Information', style: Theme.of(context).textTheme.titleLarge),
                    const SizedBox(height: 16),
                    TextField(
                      controller: _nameController,
                      decoration: const InputDecoration(labelText: 'Full Name', border: OutlineInputBorder()),
                    ),
                    const SizedBox(height: 16),
                    TextField(
                      controller: _emailController,
                      decoration: const InputDecoration(labelText: 'Email Address', border: OutlineInputBorder()),
                      keyboardType: TextInputType.emailAddress,
                    ),
                    const SizedBox(height: 24),
                    FilledButton.icon(
                      onPressed: () {
                        setState(() => _isSaved = true);
                        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Profile saved successfully!')));
                      },
                      icon: const Icon(Icons.save),
                      label: const Text('Save Profile'),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 48),

            // Feature List
            Text('App Features', style: Theme.of(context).textTheme.headlineSmall, textAlign: TextAlign.center),
            const SizedBox(height: 16),
            _buildFeatureItem(
              context,
              Icons.psychology,
              'Career Assessment',
              'Take a comprehensive RIASEC test along with your skills and interests to get AI-powered career recommendations.',
            ),
            _buildFeatureItem(
              context,
              Icons.document_scanner,
              'Resume Analysis',
              'Upload your resume text to get an ATS compatibility score and identify missing skills for your target role.',
            ),
            _buildFeatureItem(
              context,
              Icons.school,
              'Interview Preparation',
              'Receive tailored interview questions and pro tips based on your skill gaps to help you land the job.',
            ),
            _buildFeatureItem(
              context,
              Icons.map,
              'Career Roadmaps',
              'Direct integration with roadmap.sh to provide structured learning paths for your recommended careers.',
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFeatureItem(BuildContext context, IconData icon, String title, String description) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Card(
        elevation: 2,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.primary.withOpacity(0.1),
                  shape: BoxShape.circle,
                ),
                child: Icon(icon, color: Theme.of(context).colorScheme.primary, size: 28),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 4),
                    Text(description, style: TextStyle(color: Colors.grey[700], height: 1.4)),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
