import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter/foundation.dart';
import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'firebase_options.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  runApp(const ProviderScope(child: PathFinderApp()));
}

final authStateProvider = StreamProvider<User?>((ref) {
  return FirebaseAuth.instance.authStateChanges();
});

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
      recentActivity: ['Completed Assessment', ...state.recentActivity],
    );
  }

  void updateAtsScore(int score) {
    state = DashboardData(
      assessmentsTaken: state.assessmentsTaken,
      topRecommendation: state.topRecommendation,
      atsScore: score,
      recentActivity: ['Completed ATS Scan: $score%', ...state.recentActivity],
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
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF4F2EF), // Warm off-white
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFFE68C3A), // Burnt Sienna
          surface: const Color(0xFFF4F2EF),
          onSurface: const Color(0xFF213E60),
        ),
        textTheme: TextTheme(
          displayLarge: GoogleFonts.playfairDisplay(fontWeight: FontWeight.w700, color: const Color(0xFF213E60), letterSpacing: -1.5),
          displayMedium: GoogleFonts.playfairDisplay(fontWeight: FontWeight.w700, color: const Color(0xFF213E60), letterSpacing: -1.0),
          headlineMedium: GoogleFonts.playfairDisplay(fontWeight: FontWeight.w600, color: const Color(0xFF213E60)),
          titleLarge: GoogleFonts.inter(fontWeight: FontWeight.w600, color: const Color(0xFF213E60), letterSpacing: -0.5),
          bodyLarge: GoogleFonts.inter(color: const Color(0xFF4A4A4A), height: 1.6),
          bodyMedium: GoogleFonts.inter(color: const Color(0xFF4A4A4A), height: 1.5),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(0), borderSide: BorderSide.none),
          focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(0), borderSide: const BorderSide(color: Color(0xFF213E60), width: 2)),
          contentPadding: const EdgeInsets.all(20),
          labelStyle: GoogleFonts.inter(color: Colors.grey[600]),
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            backgroundColor: const Color(0xFF213E60),
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 32),
            shape: const RoundedRectangleBorder(borderRadius: BorderRadius.zero),
            textStyle: GoogleFonts.inter(fontWeight: FontWeight.w600, letterSpacing: 1),
          ),
        ),
        cardTheme: const CardThemeData(
          color: Colors.white,
          elevation: 0,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.zero),
          margin: EdgeInsets.zero,
        ),
      ),
      home: const AuthWrapper(),
    );
  }
}

class AuthWrapper extends ConsumerWidget {
  const AuthWrapper({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final authState = ref.watch(authStateProvider);
    return authState.when(
      data: (user) {
        if (user == null) return const AuthScreen();
        return const MainLayout();
      },
      loading: () => const Scaffold(body: Center(child: CircularProgressIndicator())),
      error: (e, trace) => Scaffold(body: Center(child: Text('Error: $e'))),
    );
  }
}

class AuthScreen extends StatefulWidget {
  const AuthScreen({super.key});
  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isLogin = true;
  bool _isLoading = false;

  Future<void> _submit() async {
    setState(() => _isLoading = true);
    try {
      if (_isLogin) {
        await FirebaseAuth.instance.signInWithEmailAndPassword(
          email: _emailController.text.trim(),
          password: _passwordController.text.trim(),
        );
      } else {
        await FirebaseAuth.instance.createUserWithEmailAndPassword(
          email: _emailController.text.trim(),
          password: _passwordController.text.trim(),
        );
      }
    } on FirebaseAuthException catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message ?? 'Authentication error'), backgroundColor: Colors.red));
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          // Left side: Editorial Image/Text (hidden on mobile)
          if (MediaQuery.of(context).size.width > 800)
            Expanded(
              child: Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Color(0xFF213E60), Color(0xFF94B6EF)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
                padding: const EdgeInsets.all(64),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text('PathFinder.', style: GoogleFonts.playfairDisplay(fontSize: 64, color: Colors.white, fontWeight: FontWeight.w700)).animate().fade(duration: 800.ms).slideX(begin: -0.1),
                    const SizedBox(height: 24),
                    Text('Map your career trajectory with precision AI analysis.', style: GoogleFonts.inter(fontSize: 24, color: Colors.grey[400], height: 1.5)).animate().fade(delay: 200.ms, duration: 800.ms).slideX(begin: -0.1),
                  ],
                ),
              ),
            ),
          
          // Right side: Form
          Expanded(
            child: Center(
              child: Container(
                constraints: const BoxConstraints(maxWidth: 400),
                padding: const EdgeInsets.all(32),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (MediaQuery.of(context).size.width <= 800) ...[
                      Text('PathFinder.', style: GoogleFonts.playfairDisplay(fontSize: 48, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 48),
                    ],
                    Text(_isLogin ? 'Welcome back.' : 'Begin your journey.', style: Theme.of(context).textTheme.headlineMedium),
                    const SizedBox(height: 8),
                    Text('Enter your credentials to continue.', style: Theme.of(context).textTheme.bodyLarge),
                    const SizedBox(height: 48),
                    TextField(
                      controller: _emailController,
                      decoration: const InputDecoration(labelText: 'Email Address'),
                      keyboardType: TextInputType.emailAddress,
                    ),
                    const SizedBox(height: 16),
                    TextField(
                      controller: _passwordController,
                      decoration: const InputDecoration(labelText: 'Password'),
                      obscureText: true,
                    ),
                    const SizedBox(height: 32),
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton(
                        onPressed: _isLoading ? null : _submit,
                        child: _isLoading ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)) : Text(_isLogin ? 'SIGN IN' : 'CREATE ACCOUNT'),
                      ),
                    ),
                    const SizedBox(height: 24),
                    Center(
                      child: TextButton(
                        onPressed: () => setState(() => _isLogin = !_isLogin),
                        style: TextButton.styleFrom(foregroundColor: const Color(0xFF213E60)),
                        child: Text(_isLogin ? 'Need an account? Sign up' : 'Already have an account? Sign in'),
                      ),
                    ),
                  ],
                ).animate().fade(duration: 600.ms).slideY(begin: 0.05),
              ),
            ),
          ),
        ],
      ),
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
        backgroundColor: Colors.transparent,
        elevation: 0,
        toolbarHeight: 80,
        title: Padding(
          padding: const EdgeInsets.only(left: 24.0),
          child: Text('PathFinder.', style: GoogleFonts.playfairDisplay(fontSize: 28, fontWeight: FontWeight.bold, color: const Color(0xFF213E60))),
        ),
        actions: [
          if (MediaQuery.of(context).size.width > 600) ...[
            _navButton('Dashboard', 'dashboard'),
            _navButton('Assessment', 'assessment'),
            _navButton('ATS Scan', 'resume'),
            _navButton('History', 'history'),
            _navButton('Profile', 'profile'),
          ] else ...[
            PopupMenuButton<String>(
              icon: const Icon(Icons.menu, color: Color(0xFF213E60)),
              onSelected: (val) => setState(() => currentView = val),
              itemBuilder: (context) => [
                const PopupMenuItem(value: 'dashboard', child: Text('Dashboard')),
                const PopupMenuItem(value: 'assessment', child: Text('Assessment')),
                const PopupMenuItem(value: 'resume', child: Text('ATS Scan')),
                const PopupMenuItem(value: 'history', child: Text('History')),
                const PopupMenuItem(value: 'profile', child: Text('Profile')),
              ],
            ),
          ],
          Padding(
            padding: const EdgeInsets.only(right: 24.0, left: 16.0),
            child: IconButton(
              onPressed: () => FirebaseAuth.instance.signOut(),
              icon: const Icon(Icons.logout, color: Color(0xFF213E60)),
              tooltip: 'Sign Out',
            ),
          ),
        ],
      ),
      body: AnimatedSwitcher(
        duration: const Duration(milliseconds: 300),
        child: _getView(),
      ),
    );
  }

  Widget _navButton(String title, String viewId) {
    final isSelected = currentView == viewId;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8.0),
      child: TextButton(
        onPressed: () => setState(() => currentView = viewId),
        style: TextButton.styleFrom(
          foregroundColor: isSelected ? const Color(0xFFE68C3A) : const Color(0xFF213E60),
          textStyle: GoogleFonts.inter(fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400, letterSpacing: 0.5),
        ),
        child: Text(title.toUpperCase()),
      ),
    );
  }

  Widget _getView() {
    switch (currentView) {
      case 'dashboard': return const DashboardView(key: ValueKey('dash'));
      case 'assessment': return const AssessmentView(key: ValueKey('assess'));
      case 'resume': return const ResumeAnalysisView(key: ValueKey('resume'));
      case 'history': return const HistoryView(key: ValueKey('history'));
      case 'profile': return const ProfileView(key: ValueKey('profile'));
      default: return const DashboardView(key: ValueKey('dash'));
    }
  }
}

// ---------------- DASHBOARD VIEW ----------------
class DashboardView extends StatefulWidget {
  const DashboardView({super.key});
  @override
  State<DashboardView> createState() => _DashboardViewState();
}

class _DashboardViewState extends State<DashboardView> {
  bool _isLoading = true;
  String? _errorMessage;
  int _assessmentsTaken = 0;
  String _topRecommendation = 'None';
  String _atsScore = 'N/A';
  List<String> _recentActivity = [];

  @override
  void initState() {
    super.initState();
    _fetchDashboardData();
  }

  Future<void> _fetchDashboardData() async {
    if (!mounted) return;
    setState(() { _isLoading = true; _errorMessage = null; });
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;
    final token = await user.getIdToken();
    try {
      final response = await http.get(
        Uri.parse('https://career-recommendation-app-2-08ny.onrender.com/api/history'),
        headers: {'Authorization': 'Bearer $token'},
      ).timeout(const Duration(seconds: 15));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final assessments = data['assessments'] as List;
        final resumes = data['resumes'] as List;
        
        int assessmentsCount = assessments.length;
        String topRec = assessmentsCount > 0 ? assessments[0]['recommended_career'] ?? 'None' : 'None';
        String ats = resumes.isNotEmpty ? '${resumes[0]['ats_score']}%' : 'N/A';

        List<String> activity = [];
        if (assessments.isNotEmpty) activity.add('Assessment: ${assessments[0]['recommended_career']}');
        if (resumes.isNotEmpty) activity.add('ATS Scan: ${resumes[0]['ats_score']}%');

        if (mounted) {
          setState(() {
            _assessmentsTaken = assessmentsCount;
            _topRecommendation = topRec;
            _atsScore = ats;
            _recentActivity = activity;
            _isLoading = false;
          });
        }
      } else {
        if (mounted) setState(() { _isLoading = false; _errorMessage = 'Server returned ${response.statusCode}. The backend may be starting up — please retry in a moment.'; });
      }
    } catch (e) {
      if (mounted) setState(() { _isLoading = false; _errorMessage = 'Could not connect to server. The backend may be waking up — please retry in a moment.'; });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) return const Center(child: Column(mainAxisSize: MainAxisSize.min, children: [CircularProgressIndicator(color: Color(0xFF213E60)), SizedBox(height: 16), Text('Loading dashboard...', style: TextStyle(color: Color(0xFF4A4A4A)))]));
    if (_errorMessage != null) {
      return Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 500),
          padding: const EdgeInsets.all(48),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.cloud_off, size: 64, color: Color(0xFFE68C3A)),
              const SizedBox(height: 24),
              Text('Connection Issue', style: GoogleFonts.playfairDisplay(fontSize: 28, fontWeight: FontWeight.w700, color: const Color(0xFF213E60))),
              const SizedBox(height: 16),
              Text(_errorMessage!, textAlign: TextAlign.center, style: const TextStyle(color: Color(0xFF4A4A4A), fontSize: 16, height: 1.5)),
              const SizedBox(height: 32),
              FilledButton.icon(
                onPressed: _fetchDashboardData,
                icon: const Icon(Icons.refresh),
                label: const Text('RETRY'),
              ),
            ],
          ),
        ),
      );
    }
    final user = FirebaseAuth.instance.currentUser;
    
    return Center(
      child: Container(
        constraints: const BoxConstraints(maxWidth: 1000),
        padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 48.0),
        child: ListView(
          children: [
            Text('Overview.', style: Theme.of(context).textTheme.displayMedium).animate().fade().slideY(begin: 0.1),
            const SizedBox(height: 16),
            Text('Welcome back, ${user?.displayName ?? user?.email?.split('@')[0] ?? 'User'}. Here is your trajectory.', style: Theme.of(context).textTheme.bodyLarge?.copyWith(fontSize: 18)).animate().fade(delay: 100.ms).slideY(begin: 0.1),
            const SizedBox(height: 64),
            
            // Metrics Grid
            LayoutBuilder(
              builder: (context, constraints) {
                final isWide = constraints.maxWidth > 600;
                return Flex(
                  direction: isWide ? Axis.horizontal : Axis.vertical,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(flex: isWide ? 1 : 0, child: _MetricCard(title: 'Assessments', value: _assessmentsTaken.toString())),
                    if (isWide) const SizedBox(width: 32) else const SizedBox(height: 32),
                    Expanded(flex: isWide ? 1 : 0, child: _MetricCard(title: 'Top Match', value: _topRecommendation)),
                    if (isWide) const SizedBox(width: 32) else const SizedBox(height: 32),
                    Expanded(flex: isWide ? 1 : 0, child: _MetricCard(title: 'ATS Score', value: _atsScore)),
                  ],
                );
              }
            ).animate().fade(delay: 200.ms).slideY(begin: 0.1),
            
            const SizedBox(height: 80),
            Text('Recent Activity.', style: Theme.of(context).textTheme.headlineMedium).animate().fade(delay: 300.ms),
            const SizedBox(height: 24),
            if (_recentActivity.isEmpty)
              Text("No recent activity recorded.", style: TextStyle(color: Colors.grey[500], fontStyle: FontStyle.italic)).animate().fade(delay: 400.ms)
            else
              ..._recentActivity.map((act) => 
                Container(
                  margin: const EdgeInsets.only(bottom: 16),
                  decoration: const BoxDecoration(border: Border(left: BorderSide(color: Color(0xFFE68C3A), width: 4))),
                  padding: const EdgeInsets.all(24),
                  color: Colors.white,
                  child: Row(
                    children: [
                      const Icon(Icons.arrow_right_alt, color: Color(0xFF213E60)),
                      const SizedBox(width: 16),
                      Expanded(child: Text(act, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500))),
                    ],
                  ),
                ).animate().fade(delay: 400.ms).slideX(begin: 0.05)
              ).toList(),
          ],
        ),
      ),
    );
  }
}

class _MetricCard extends StatelessWidget {
  final String title;
  final String value;
  const _MetricCard({required this.title, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(32),
      color: Colors.white,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title.toUpperCase(), style: GoogleFonts.inter(color: Colors.grey[500], fontWeight: FontWeight.w600, letterSpacing: 1.5, fontSize: 12)),
          const SizedBox(height: 16),
          Text(value, style: GoogleFonts.playfairDisplay(fontSize: 32, fontWeight: FontWeight.w700, color: const Color(0xFF213E60), height: 1.1)),
        ],
      ),
    );
  }
}

// ---------------- ASSESSMENT VIEW ----------------
class AssessmentView extends StatefulWidget {
  const AssessmentView({super.key});
  @override
  State<AssessmentView> createState() => _AssessmentViewState();
}

class _AssessmentViewState extends State<AssessmentView> {
  final _formKey = GlobalKey<FormState>();
  bool isLoading = false;
  Map<String, dynamic>? result;

  final _ageController = TextEditingController(text: '24');
  String _education = "Bachelor's";
  final _skillsController = TextEditingController(text: 'Python, SQL, Communication');
  final _interestsController = TextEditingController(text: 'Data, Analysis, Teamwork');

  double _scoreR = 5, _scoreI = 5, _scoreA = 5, _scoreS = 5, _scoreE = 5, _scoreC = 5;

  Future<void> _submitAssessment() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => isLoading = true);
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      setState(() => isLoading = false);
      return;
    }
    final token = await user.getIdToken();

    try {
      final response = await http.post(
        Uri.parse('https://career-recommendation-app-2-08ny.onrender.com/api/recommend'),
        headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer $token'},
        body: jsonEncode({
          "age": int.tryParse(_ageController.text) ?? 24,
          "education": _education,
          "skills": _skillsController.text.split(',').map((e) => e.trim()).where((e) => e.isNotEmpty).toList(),
          "interests": _interestsController.text.split(',').map((e) => e.trim()).where((e) => e.isNotEmpty).toList(),
          "riasec_scores": [_scoreR, _scoreI, _scoreA, _scoreS, _scoreE, _scoreC]
        }),
      );

      if (response.statusCode == 200) {
        setState(() => result = jsonDecode(response.body));
      } else {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: ${response.statusCode}')));
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Connection failed.')));
    } finally {
      if (mounted) setState(() => isLoading = false);
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
            Text(value.toInt().toString(), style: const TextStyle(color: Color(0xFF213E60), fontWeight: FontWeight.bold)),
          ],
        ),
        const SizedBox(height: 8),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            activeTrackColor: const Color(0xFF213E60),
            inactiveTrackColor: Colors.grey[300],
            thumbColor: const Color(0xFFE68C3A),
            overlayColor: const Color(0xFFE68C3A).withValues(alpha: 0.2),
          ),
          child: Slider(value: value, min: 0, max: 10, divisions: 10, onChanged: onChanged),
        ),
        const SizedBox(height: 16),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    if (result != null) return _buildResult(result!);

    return Center(
      child: Container(
        constraints: const BoxConstraints(maxWidth: 1000),
        padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 48.0),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              Text('Assessment.', style: Theme.of(context).textTheme.displayMedium).animate().fade().slideY(begin: 0.1),
              const SizedBox(height: 16),
              Text('Detail your background and preferences to generate a personalized career trajectory.', style: Theme.of(context).textTheme.bodyLarge?.copyWith(fontSize: 18)).animate().fade(delay: 100.ms).slideY(begin: 0.1),
              const SizedBox(height: 64),
              
              LayoutBuilder(
                builder: (context, constraints) {
                  final isWide = constraints.maxWidth > 700;
                  final col1 = Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Profile Data', style: Theme.of(context).textTheme.headlineMedium),
                      const SizedBox(height: 32),
                      TextFormField(
                        controller: _ageController,
                        decoration: const InputDecoration(labelText: 'Age'),
                        keyboardType: TextInputType.number,
                      ),
                      const SizedBox(height: 24),
                      DropdownButtonFormField<String>(
                        value: _education,
                        decoration: const InputDecoration(labelText: 'Highest Education'),
                        icon: const Icon(Icons.keyboard_arrow_down),
                        items: ['High School', "Bachelor's", "Master's", 'PhD', 'Self-Taught'].map((e) => DropdownMenuItem(value: e, child: Text(e))).toList(),
                        onChanged: (val) => setState(() => _education = val!),
                      ),
                      const SizedBox(height: 24),
                      TextFormField(
                        controller: _skillsController,
                        decoration: const InputDecoration(labelText: 'Skills (comma separated)'),
                      ),
                      const SizedBox(height: 24),
                      TextFormField(
                        controller: _interestsController,
                        decoration: const InputDecoration(labelText: 'Interests (comma separated)'),
                      ),
                    ],
                  );
                  
                  final col2 = Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('RIASEC Profile', style: Theme.of(context).textTheme.headlineMedium),
                      const SizedBox(height: 32),
                      _buildSlider('Realistic', _scoreR, (v) => setState(() => _scoreR = v)),
                      _buildSlider('Investigative', _scoreI, (v) => setState(() => _scoreI = v)),
                      _buildSlider('Artistic', _scoreA, (v) => setState(() => _scoreA = v)),
                      _buildSlider('Social', _scoreS, (v) => setState(() => _scoreS = v)),
                      _buildSlider('Enterprising', _scoreE, (v) => setState(() => _scoreE = v)),
                      _buildSlider('Conventional', _scoreC, (v) => setState(() => _scoreC = v)),
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
                }
              ).animate().fade(delay: 200.ms).slideY(begin: 0.1),
              
              const SizedBox(height: 64),
              Center(
                child: SizedBox(
                  width: 300,
                  child: FilledButton(
                    onPressed: isLoading ? null : _submitAssessment,
                    child: isLoading ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)) : const Text('ANALYZE PROFILE'),
                  ),
                ),
              ).animate().fade(delay: 300.ms),
              const SizedBox(height: 64),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildResult(Map<String, dynamic> res) {
    final pred = res['prediction'];
    return Center(
      child: Container(
        constraints: const BoxConstraints(maxWidth: 800),
        padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 48.0),
        child: ListView(
          children: [
            Text('Analysis Complete.', style: Theme.of(context).textTheme.displayMedium).animate().fade().slideY(begin: 0.1),
            const SizedBox(height: 64),
            
            Container(
              padding: const EdgeInsets.all(48),
              color: Colors.white,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('PRIMARY MATCH', style: GoogleFonts.inter(color: const Color(0xFFE68C3A), fontWeight: FontWeight.w700, letterSpacing: 2, fontSize: 12)),
                  const SizedBox(height: 16),
                  Text(pred['Recommended_Career'], style: GoogleFonts.playfairDisplay(fontSize: 48, fontWeight: FontWeight.w700, color: const Color(0xFF213E60), height: 1.1)),
                  const SizedBox(height: 24),
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                        color: const Color(0xFF213E60),
                        child: Text('Confidence: ${(pred['Recommendation_Score'] * 100).toStringAsFixed(1)}%', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                      ),
                      const SizedBox(width: 16),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                        decoration: BoxDecoration(border: Border.all(color: Colors.grey[300]!)),
                        child: Text('Cluster #${pred['Unsupervised_Cluster']}', style: const TextStyle(fontWeight: FontWeight.bold)),
                      ),
                    ],
                  ),
                  const SizedBox(height: 48),
                  const Divider(),
                  const SizedBox(height: 48),
                  
                  Text('STRONG ALTERNATIVES', style: GoogleFonts.inter(color: Colors.grey[500], fontWeight: FontWeight.w700, letterSpacing: 2, fontSize: 12)),
                  const SizedBox(height: 24),
                  for (var alt in pred['Top_3_Careers'])
                    Padding(
                      padding: const EdgeInsets.only(bottom: 16.0),
                      child: Row(
                        children: [
                          const Icon(Icons.arrow_right_alt, color: Color(0xFFE68C3A)),
                          const SizedBox(width: 16),
                          Expanded(child: Text('${alt['career']}', style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w500))),
                          Text('${(alt['score'] * 100).toStringAsFixed(1)}%', style: TextStyle(color: Colors.grey[500], fontSize: 18)),
                        ],
                      ),
                    ),
                ],
              ),
            ).animate().fade(delay: 200.ms).slideY(begin: 0.1),
            
            const SizedBox(height: 48),
            Center(
              child: TextButton.icon(
                onPressed: () => setState(() => result = null),
                icon: const Icon(Icons.refresh, color: Color(0xFF213E60)),
                label: const Text('NEW ASSESSMENT', style: TextStyle(color: Color(0xFF213E60), letterSpacing: 1, fontWeight: FontWeight.bold)),
              ),
            ).animate().fade(delay: 300.ms),
          ],
        ),
      ),
    );
  }
}

// ---------------- RESUME ANALYSIS VIEW ----------------
class ResumeAnalysisView extends StatefulWidget {
  const ResumeAnalysisView({super.key});
  @override
  State<ResumeAnalysisView> createState() => _ResumeAnalysisViewState();
}

class _ResumeAnalysisViewState extends State<ResumeAnalysisView> {
  final _formKey = GlobalKey<FormState>();
  bool isLoading = false;
  Map<String, dynamic>? result;

  final _resumeController = TextEditingController();
  final _targetCareerController = TextEditingController(text: 'Data Scientist');

  Future<void> _analyzeResume() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => isLoading = true);
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      setState(() => isLoading = false);
      return;
    }
    final token = await user.getIdToken();

    try {
      final response = await http.post(
        Uri.parse('https://career-recommendation-app-2-08ny.onrender.com/api/resume/analyze'),
        headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer $token'},
        body: jsonEncode({"resume_text": _resumeController.text, "target_career": _targetCareerController.text}),
      );

      if (response.statusCode == 200) {
        setState(() => result = jsonDecode(response.body));
      } else {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: ${response.statusCode}')));
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Connection failed.')));
    } finally {
      if (mounted) setState(() => isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (result != null) return _buildResult(result!);

    return Center(
      child: Container(
        constraints: const BoxConstraints(maxWidth: 800),
        padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 48.0),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              Text('ATS Scan.', style: Theme.of(context).textTheme.displayMedium).animate().fade().slideY(begin: 0.1),
              const SizedBox(height: 16),
              Text('Upload your resume text to evaluate compatibility against your target role.', style: Theme.of(context).textTheme.bodyLarge?.copyWith(fontSize: 18)).animate().fade(delay: 100.ms).slideY(begin: 0.1),
              const SizedBox(height: 64),
              
              TextFormField(
                controller: _targetCareerController,
                decoration: const InputDecoration(labelText: 'Target Role (e.g. Data Scientist, UX Designer)'),
                validator: (val) => val!.isEmpty ? 'Required' : null,
              ).animate().fade(delay: 200.ms).slideY(begin: 0.1),
              const SizedBox(height: 32),
              
              TextFormField(
                controller: _resumeController,
                decoration: const InputDecoration(labelText: 'Paste Resume Content Here', alignLabelWithHint: true),
                maxLines: 12,
                validator: (val) => val!.isEmpty ? 'Required' : null,
              ).animate().fade(delay: 300.ms).slideY(begin: 0.1),
              const SizedBox(height: 48),
              
              Center(
                child: SizedBox(
                  width: 300,
                  child: FilledButton(
                    onPressed: isLoading ? null : _analyzeResume,
                    child: isLoading ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)) : const Text('INITIATE SCAN'),
                  ),
                ),
              ).animate().fade(delay: 400.ms),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildResult(Map<String, dynamic> res) {
    final analysis = res['analysis'];
    final atsScore = analysis['ats_score'] as double;
    
    return Center(
      child: Container(
        constraints: const BoxConstraints(maxWidth: 800),
        padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 48.0),
        child: ListView(
          children: [
            Text('Scan Complete.', style: Theme.of(context).textTheme.displayMedium).animate().fade().slideY(begin: 0.1),
            const SizedBox(height: 64),
            
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
                            Text('OVERALL MATCH', style: GoogleFonts.inter(color: Colors.grey[500], fontWeight: FontWeight.w700, letterSpacing: 2, fontSize: 12)),
                            const SizedBox(height: 16),
                            Text(analysis['recommendation'], style: Theme.of(context).textTheme.headlineMedium?.copyWith(height: 1.3)),
                          ],
                        ),
                      ),
                      const SizedBox(width: 48),
                      Container(
                        padding: const EdgeInsets.all(32),
                        decoration: BoxDecoration(
                          border: Border.all(color: atsScore > 75 ? const Color(0xFF94B6EF) : (atsScore > 50 ? const Color(0xFFE68C3A) : Colors.red), width: 4),
                          shape: BoxShape.circle,
                        ),
                        child: Text('${atsScore.toInt()}%', style: GoogleFonts.playfairDisplay(fontSize: 48, fontWeight: FontWeight.bold, color: const Color(0xFF213E60))),
                      )
                    ],
                  ),
                  const SizedBox(height: 64),
                  
                  LayoutBuilder(
                    builder: (context, constraints) {
                      final isWide = constraints.maxWidth > 500;
                      final verified = Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('VERIFIED SKILLS', style: GoogleFonts.inter(color: const Color(0xFF94B6EF), fontWeight: FontWeight.w700, letterSpacing: 2, fontSize: 12)),
                          const SizedBox(height: 24),
                          Wrap(
                            spacing: 8, runSpacing: 8,
                            children: (analysis['skills_found'] as List).map<Widget>((s) => Container(
                              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                              color: const Color(0xFF94B6EF).withValues(alpha: 0.1),
                              child: Text(s, style: const TextStyle(color: Color(0xFF94B6EF), fontWeight: FontWeight.w600)),
                            )).toList(),
                          )
                        ],
                      );
                      
                      final missing = Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('MISSING SKILLS', style: GoogleFonts.inter(color: const Color(0xFFE68C3A), fontWeight: FontWeight.w700, letterSpacing: 2, fontSize: 12)),
                          const SizedBox(height: 24),
                          Wrap(
                            spacing: 8, runSpacing: 8,
                            children: (analysis['skills_missing'] as List).map<Widget>((s) => Container(
                              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                              color: const Color(0xFFE68C3A).withValues(alpha: 0.1),
                              child: Text(s, style: const TextStyle(color: Color(0xFFE68C3A), fontWeight: FontWeight.w600)),
                            )).toList(),
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
                    }
                  ),
                  
                  if (res['interview_prep'] != null) ...[
                    const SizedBox(height: 64),
                    const Divider(),
                    const SizedBox(height: 64),
                    Text('INTERVIEW PREPARATION', style: GoogleFonts.inter(color: Colors.grey[500], fontWeight: FontWeight.w700, letterSpacing: 2, fontSize: 12)),
                    const SizedBox(height: 32),
                    Text('Tailored Questions', style: Theme.of(context).textTheme.titleLarge),
                    const SizedBox(height: 16),
                    ...(res['interview_prep']['interview_questions'] as List).map((q) => Padding(
                      padding: const EdgeInsets.only(bottom: 12.0),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Padding(padding: EdgeInsets.only(top: 8), child: Icon(Icons.circle, size: 8, color: Color(0xFF213E60))),
                          const SizedBox(width: 16),
                          Expanded(child: Text(q, style: const TextStyle(fontSize: 16, height: 1.5))),
                        ],
                      ),
                    )).toList(),
                    
                    const SizedBox(height: 48),
                    Center(
                      child: OutlinedButton.icon(
                        onPressed: () async {
                          final url = Uri.parse(res['interview_prep']['roadmap_url']);
                          if (await canLaunchUrl(url)) await launchUrl(url);
                        },
                        icon: const Icon(Icons.map, color: Color(0xFF213E60)),
                        label: const Text('VIEW ROADMAP', style: TextStyle(color: Color(0xFF213E60), letterSpacing: 1, fontWeight: FontWeight.bold)),
                        style: OutlinedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 24),
                          side: const BorderSide(color: Color(0xFF213E60), width: 2),
                          shape: const RoundedRectangleBorder(borderRadius: BorderRadius.zero),
                        )
                      ),
                    ),
                  ]
                ],
              ),
            ).animate().fade(delay: 200.ms).slideY(begin: 0.1),
            
            const SizedBox(height: 48),
            Center(
              child: TextButton.icon(
                onPressed: () => setState(() => result = null),
                icon: const Icon(Icons.refresh, color: Color(0xFF213E60)),
                label: const Text('NEW SCAN', style: TextStyle(color: Color(0xFF213E60), letterSpacing: 1, fontWeight: FontWeight.bold)),
              ),
            ).animate().fade(delay: 300.ms),
          ],
        ),
      ),
    );
  }
}

// ---------------- HISTORY VIEW ----------------
class HistoryView extends StatefulWidget {
  const HistoryView({super.key});
  @override
  State<HistoryView> createState() => _HistoryViewState();
}

class _HistoryViewState extends State<HistoryView> {
  bool _isLoading = true;
  String? _errorMessage;
  List<dynamic> _assessments = [];
  List<dynamic> _resumes = [];

  @override
  void initState() {
    super.initState();
    _fetchHistory();
  }

  Future<void> _fetchHistory() async {
    if (!mounted) return;
    setState(() { _isLoading = true; _errorMessage = null; });
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;
    final token = await user.getIdToken();
    try {
      final response = await http.get(
        Uri.parse('https://career-recommendation-app-2-08ny.onrender.com/api/history'),
        headers: {'Authorization': 'Bearer $token'},
      ).timeout(const Duration(seconds: 15));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (mounted) {
          setState(() {
            _assessments = data['assessments'];
            _resumes = data['resumes'];
            _isLoading = false;
          });
        }
      } else {
        if (mounted) setState(() { _isLoading = false; _errorMessage = 'Server returned ${response.statusCode}. The backend may be starting up — please retry in a moment.'; });
      }
    } catch (e) {
      if (mounted) setState(() { _isLoading = false; _errorMessage = 'Could not connect to server. The backend may be waking up — please retry in a moment.'; });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) return const Center(child: Column(mainAxisSize: MainAxisSize.min, children: [CircularProgressIndicator(color: Color(0xFF213E60)), SizedBox(height: 16), Text('Loading history...', style: TextStyle(color: Color(0xFF4A4A4A)))]));
    if (_errorMessage != null) {
      return Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 500),
          padding: const EdgeInsets.all(48),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.cloud_off, size: 64, color: Color(0xFFE68C3A)),
              const SizedBox(height: 24),
              Text('Connection Issue', style: GoogleFonts.playfairDisplay(fontSize: 28, fontWeight: FontWeight.w700, color: const Color(0xFF213E60))),
              const SizedBox(height: 16),
              Text(_errorMessage!, textAlign: TextAlign.center, style: const TextStyle(color: Color(0xFF4A4A4A), fontSize: 16, height: 1.5)),
              const SizedBox(height: 32),
              FilledButton.icon(
                onPressed: _fetchHistory,
                icon: const Icon(Icons.refresh),
                label: const Text('RETRY'),
              ),
            ],
          ),
        ),
      );
    }
    
    return Center(
      child: Container(
        constraints: const BoxConstraints(maxWidth: 800),
        padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 48.0),
        child: ListView(
          children: [
            Text('History.', style: Theme.of(context).textTheme.displayMedium).animate().fade().slideY(begin: 0.1),
            const SizedBox(height: 16),
            Text('A complete timeline of your career mapping.', style: Theme.of(context).textTheme.bodyLarge?.copyWith(fontSize: 18)).animate().fade(delay: 100.ms).slideY(begin: 0.1),
            const SizedBox(height: 64),
            
            Text('ASSESSMENTS', style: GoogleFonts.inter(color: Colors.grey[500], fontWeight: FontWeight.w700, letterSpacing: 2, fontSize: 12)).animate().fade(delay: 200.ms),
            const SizedBox(height: 24),
            if (_assessments.isEmpty) 
              Text('No assessments on record.', style: TextStyle(color: Colors.grey[500], fontStyle: FontStyle.italic)).animate().fade(delay: 200.ms)
            else
              ..._assessments.map((a) => Container(
                margin: const EdgeInsets.only(bottom: 16),
                padding: const EdgeInsets.all(32),
                color: Colors.white,
                child: Row(
                  children: [
                    Container(padding: const EdgeInsets.all(16), color: const Color(0xFFF4F2EF), child: const Icon(Icons.psychology, color: Color(0xFF213E60))),
                    const SizedBox(width: 24),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(a['recommended_career'] ?? 'Unknown', style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                          const SizedBox(height: 4),
                          Text('Confidence: ${((a['recommendation_score'] ?? 0) * 100).toStringAsFixed(1)}%', style: TextStyle(color: Colors.grey[600])),
                        ],
                      ),
                    ),
                    Text(a['created_at']?.split('T')[0] ?? '', style: TextStyle(color: Colors.grey[400], fontWeight: FontWeight.w500)),
                  ],
                ),
              ).animate().fade(delay: 250.ms).slideX(begin: 0.05)).toList(),
              
            const SizedBox(height: 64),
            Text('ATS SCANS', style: GoogleFonts.inter(color: Colors.grey[500], fontWeight: FontWeight.w700, letterSpacing: 2, fontSize: 12)).animate().fade(delay: 300.ms),
            const SizedBox(height: 24),
            if (_resumes.isEmpty) 
              Text('No resumes on record.', style: TextStyle(color: Colors.grey[500], fontStyle: FontStyle.italic)).animate().fade(delay: 300.ms)
            else
              ..._resumes.map((r) => Container(
                margin: const EdgeInsets.only(bottom: 16),
                padding: const EdgeInsets.all(32),
                color: Colors.white,
                child: Row(
                  children: [
                    Container(padding: const EdgeInsets.all(16), color: const Color(0xFFF4F2EF), child: const Icon(Icons.document_scanner, color: Color(0xFF213E60))),
                    const SizedBox(width: 24),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('ATS Compatibility', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                          const SizedBox(height: 4),
                          Text('Score: ${r['ats_score']}%', style: TextStyle(color: Colors.grey[600])),
                        ],
                      ),
                    ),
                    Text(r['created_at']?.split('T')[0] ?? '', style: TextStyle(color: Colors.grey[400], fontWeight: FontWeight.w500)),
                  ],
                ),
              ).animate().fade(delay: 350.ms).slideX(begin: 0.05)).toList(),
          ],
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
  final _nameController = TextEditingController(text: FirebaseAuth.instance.currentUser?.displayName ?? '');
  final _emailController = TextEditingController(text: FirebaseAuth.instance.currentUser?.email ?? '');

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Container(
        constraints: const BoxConstraints(maxWidth: 800),
        padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 48.0),
        child: ListView(
          children: [
            Text('Profile.', style: Theme.of(context).textTheme.displayMedium).animate().fade().slideY(begin: 0.1),
            const SizedBox(height: 16),
            Text('Manage your account information.', style: Theme.of(context).textTheme.bodyLarge?.copyWith(fontSize: 18)).animate().fade(delay: 100.ms).slideY(begin: 0.1),
            const SizedBox(height: 64),

            Container(
              color: Colors.white,
              padding: const EdgeInsets.all(48),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('ACCOUNT DETAILS', style: GoogleFonts.inter(color: Colors.grey[500], fontWeight: FontWeight.w700, letterSpacing: 2, fontSize: 12)),
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
                  ),
                  const SizedBox(height: 48),
                  FilledButton(
                    onPressed: () {
                      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Profile saved successfully.')));
                    },
                    child: const Text('SAVE CHANGES'),
                  ),
                ],
              ),
            ).animate().fade(delay: 200.ms).slideY(begin: 0.1),
          ],
        ),
      ),
    );
  }
}
