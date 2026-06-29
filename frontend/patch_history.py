import re

with open("lib/main.dart", "r") as f:
    content = f.read()

# 1. Update MainLayout
old_nav = """          TextButton(
            onPressed: () => setState(() => currentView = 'profile'),
            child: const Text('Profile'),
          ),
          IconButton("""
new_nav = """          TextButton(
            onPressed: () => setState(() => currentView = 'history'),
            child: const Text('History'),
          ),
          TextButton(
            onPressed: () => setState(() => currentView = 'profile'),
            child: const Text('Profile'),
          ),
          IconButton("""
content = content.replace(old_nav, new_nav)

old_body = """      body: currentView == 'dashboard' 
          ? const DashboardView() 
          : currentView == 'assessment' 
              ? const AssessmentView() 
              : currentView == 'resume'
                  ? const ResumeAnalysisView()
                  : const ProfileView(),"""
new_body = """      body: currentView == 'dashboard' 
          ? const DashboardView() 
          : currentView == 'assessment' 
              ? const AssessmentView() 
              : currentView == 'resume'
                  ? const ResumeAnalysisView()
                  : currentView == 'history'
                      ? const HistoryView()
                      : const ProfileView(),"""
content = content.replace(old_body, new_body)

# 2. Add HistoryView
history_view = """
// ---------------- HISTORY VIEW ----------------
class HistoryView extends StatefulWidget {
  const HistoryView({super.key});
  @override
  State<HistoryView> createState() => _HistoryViewState();
}

class _HistoryViewState extends State<HistoryView> {
  bool _isLoading = true;
  List<dynamic> _assessments = [];
  List<dynamic> _resumes = [];

  @override
  void initState() {
    super.initState();
    _fetchHistory();
  }

  Future<void> _fetchHistory() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;
    final token = await user.getIdToken();
    try {
      final response = await http.get(
        Uri.parse('https://career-recommendation-app-2-08ny.onrender.com/api/history'),
        headers: {'Authorization': 'Bearer $token'},
      );
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (mounted) {
          setState(() {
            _assessments = data['assessments'];
            _resumes = data['resumes'];
            _isLoading = false;
          });
        }
      }
    } catch (e) {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) return const Center(child: CircularProgressIndicator());
    return Center(
      child: Container(
        constraints: const BoxConstraints(maxWidth: 800),
        padding: const EdgeInsets.all(24.0),
        child: ListView(
          children: [
            const Icon(Icons.history, size: 48, color: Color(0xFF6C63FF)),
            const SizedBox(height: 16),
            Text('Your Progress History', style: Theme.of(context).textTheme.headlineMedium, textAlign: TextAlign.center),
            const SizedBox(height: 32),
            Text('Career Assessments', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 16),
            if (_assessments.isEmpty) const Text('No assessments taken yet.'),
            ..._assessments.map((a) => Card(
              child: ListTile(
                leading: const Icon(Icons.psychology, color: Colors.blue),
                title: Text(a['recommended_career'] ?? 'Unknown Career', style: const TextStyle(fontWeight: FontWeight.bold)),
                subtitle: Text('Confidence: ${((a['recommendation_score'] ?? 0) * 100).toStringAsFixed(1)}%'),
                trailing: Text(a['created_at']?.split('T')[0] ?? ''),
              )
            )),
            const SizedBox(height: 32),
            Text('Resume Scans', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 16),
            if (_resumes.isEmpty) const Text('No resumes scanned yet.'),
            ..._resumes.map((r) => Card(
              child: ListTile(
                leading: const Icon(Icons.document_scanner, color: Colors.green),
                title: Text('ATS Score: ${r['ats_score']}%', style: const TextStyle(fontWeight: FontWeight.bold)),
                trailing: Text(r['created_at']?.split('T')[0] ?? ''),
              )
            )),
          ],
        )
      )
    );
  }
}

"""
content += history_view

# 3. Update DashboardView to fetch dynamic history data
old_dashboard = """// ---------------- DASHBOARD VIEW ----------------
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
}"""
new_dashboard = """// ---------------- DASHBOARD VIEW ----------------
class DashboardView extends StatefulWidget {
  const DashboardView({super.key});

  @override
  State<DashboardView> createState() => _DashboardViewState();
}

class _DashboardViewState extends State<DashboardView> {
  bool _isLoading = true;
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
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;
    final token = await user.getIdToken();
    try {
      final response = await http.get(
        Uri.parse('https://career-recommendation-app-2-08ny.onrender.com/api/history'),
        headers: {'Authorization': 'Bearer $token'},
      );
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final assessments = data['assessments'] as List;
        final resumes = data['resumes'] as List;
        
        int assessmentsCount = assessments.length;
        String topRec = 'None';
        if (assessmentsCount > 0) {
          topRec = assessments[0]['recommended_career'] ?? 'None';
        }
        
        String ats = 'N/A';
        if (resumes.isNotEmpty) {
          ats = '${resumes[0]['ats_score']}%';
        }

        List<String> activity = [];
        if (assessments.isNotEmpty) activity.add('Completed Career Assessment: ${assessments[0]['recommended_career']}');
        if (resumes.isNotEmpty) activity.add('Completed ATS Scan: ${resumes[0]['ats_score']}%');

        if (mounted) {
          setState(() {
            _assessmentsTaken = assessmentsCount;
            _topRecommendation = topRec;
            _atsScore = ats;
            _recentActivity = activity;
            _isLoading = false;
          });
        }
      }
    } catch (e) {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) return const Center(child: CircularProgressIndicator());
    final user = FirebaseAuth.instance.currentUser;
    
    return Center(
      child: Container(
        constraints: const BoxConstraints(maxWidth: 900),
        padding: const EdgeInsets.all(24.0),
        child: ListView(
          children: [
            Text('Welcome back, ${user?.displayName ?? 'User'}', style: Theme.of(context).textTheme.headlineMedium),
            const SizedBox(height: 8),
            Text('Here is your AI-powered career overview.', style: Theme.of(context).textTheme.bodyLarge?.copyWith(color: Colors.grey[700])),
            const SizedBox(height: 32),
            Row(
              children: [
                Expanded(child: _DashboardCard(title: 'Assessments Taken', value: _assessmentsTaken.toString(), icon: Icons.assignment_turned_in)),
                const SizedBox(width: 16),
                Expanded(child: _DashboardCard(title: 'Top Recommendation', value: _topRecommendation, icon: Icons.star)),
                const SizedBox(width: 16),
                Expanded(child: _DashboardCard(title: 'ATS Resume Score', value: _atsScore, icon: Icons.document_scanner)),
              ],
            ),
            const SizedBox(height: 32),
            Text('Recent Activity', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 16),
            if (_recentActivity.isEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 32.0),
                child: Center(child: Text("No recent activity.", style: TextStyle(color: Colors.grey[600]))),
              )
            else
              ListView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: _recentActivity.length,
                itemBuilder: (context, index) {
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 8.0),
                    child: ListTile(
                      leading: const CircleAvatar(child: Icon(Icons.analytics)),
                      title: Text(_recentActivity[index]),
                      subtitle: const Text('Recently'),
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
}"""

content = content.replace(old_dashboard, new_dashboard)

with open("lib/main.dart", "w") as f:
    f.write(content)
