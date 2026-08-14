import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:file_picker/file_picker.dart';
import '../../common/app_scaffold.dart';
import '../../providers/resume_provider.dart';
import '../../providers/history_provider.dart';
import '../../../domain/models/resume_model.dart';
import '../../../core/theme/app_theme.dart';

enum InputMode { fileUpload, pasteText }

class ResumeScreen extends ConsumerStatefulWidget {
  const ResumeScreen({super.key});

  @override
  ConsumerState<ResumeScreen> createState() => _ResumeScreenState();
}

class _ResumeScreenState extends ConsumerState<ResumeScreen> {
  final _formKey = GlobalKey<FormState>();
  final _resumeController = TextEditingController();
  final _targetCareerController = TextEditingController();
  final _geminiKeyController = TextEditingController();

  InputMode _inputMode = InputMode.fileUpload;
  List<int>? _selectedFileBytes;
  String? _selectedFileName;
  String? _selectedMimeType;
  final bool _showGeminiKeyInput = false;

  final List<String> _suggestedRoles = [
    'Software Engineer',
    'Data Scientist',
    'UX Designer',
    'Data Analyst',
    'Product Manager',
    'Cybersecurity Analyst',
    'DevOps Engineer',
  ];

  @override
  void dispose() {
    _resumeController.dispose();
    _targetCareerController.dispose();
    _geminiKeyController.dispose();
    super.dispose();
  }

  Future<void> _pickFile() async {
    try {
      final result = await FilePicker.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['pdf', 'png', 'jpg', 'jpeg', 'webp'],
        withData: true,
      );

      if (result != null && result.files.isNotEmpty) {
        final file = result.files.first;
        final ext = file.extension?.toLowerCase() ?? '';
        String mime = 'application/pdf';
        if (ext == 'png') mime = 'image/png';
        if (ext == 'jpg' || ext == 'jpeg') mime = 'image/jpeg';
        if (ext == 'webp') mime = 'image/webp';

        setState(() {
          _selectedFileBytes = file.bytes;
          _selectedFileName = file.name;
          _selectedMimeType = mime;
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error selecting file: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  void _clearSelectedFile() {
    setState(() {
      _selectedFileBytes = null;
      _selectedFileName = null;
      _selectedMimeType = null;
    });
  }

  void _resetForm() {
    setState(() {
      _resumeController.clear();
      _targetCareerController.clear();
      _selectedFileBytes = null;
      _selectedFileName = null;
      _selectedMimeType = null;
    });
    ref.read(resumeProvider.notifier).reset();
  }

  void _loadSampleData() {
    setState(() {
      _inputMode = InputMode.pasteText;
      _targetCareerController.text = 'Software Engineer';
      _resumeController.text = '''Alex Johnson
Software Engineer | Full-Stack Developer
Email: alex.johnson@email.com | GitHub: github.com/alexj

SUMMARY
Dedicated Software Engineer with 3+ years of experience building scalable web applications using Python, JavaScript, Flask, PostgreSQL, and Git. Passionate about clean code, API architecture, and CI/CD automation.

TECHNICAL SKILLS
- Languages: Python, JavaScript, SQL, HTML/CSS
- Frameworks & Tools: Flask, React, Docker, Git, RESTful APIs, PostgreSQL
- Concepts: Object-Oriented Design, Agile/Scrum, Unit Testing

EXPERIENCE
Software Developer | TechCorp Solutions (2022 - Present)
- Designed and implemented RESTful microservices in Flask handling 100k+ daily API requests.
- Optimized database queries, reducing average response latency by 35%.
- Integrated Docker containerization into existing deployment workflows.

EDUCATION
B.S. in Computer Science | State University (2018 - 2022)
''';
    });
  }

  Future<void> _analyzeResume() async {
    if (!_formKey.currentState!.validate()) return;

    if (_inputMode == InputMode.fileUpload && _selectedFileBytes == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select a PDF or Image resume file (or switch to Paste Text).'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    if (_inputMode == InputMode.pasteText &&
        _resumeController.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please enter or paste your resume text.'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    final success = await ref.read(resumeProvider.notifier).analyzeResume(
          targetCareer: _targetCareerController.text.trim(),
          resumeText: _inputMode == InputMode.pasteText
              ? _resumeController.text.trim()
              : null,
          fileBytes: _inputMode == InputMode.fileUpload ? _selectedFileBytes : null,
          fileName: _selectedFileName,
          mimeType: _selectedMimeType,
          geminiApiKey: _geminiKeyController.text.trim().isNotEmpty
              ? _geminiKeyController.text.trim()
              : null,
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
          constraints: const BoxConstraints(maxWidth: 850),
          padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 36.0),
          child: state.result != null
              ? _buildResultView(context, state.result!)
              : Form(
                  key: _formKey,
                  child: ListView(
                    children: [
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text('ATS & AI Resume Scanner.',
                                        style: Theme.of(context).textTheme.displayMedium)
                                    .animate()
                                    .fade()
                                    .slideY(begin: 0.1),
                                const SizedBox(height: 12),
                                Text(
                                  'Upload your PDF/Image resume or paste content to get deep AI-powered feedback, ATS score, missing skills, and interview questions.',
                                  style: Theme.of(context)
                                      .textTheme
                                      .bodyLarge
                                      ?.copyWith(fontSize: 16, height: 1.5),
                                ).animate().fade(delay: 100.ms).slideY(begin: 0.1),
                              ],
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 36),

                      if (state.errorMessage != null) ...[
                        Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: Colors.red.withValues(alpha: 0.08),
                            border: Border.all(color: Colors.red.withValues(alpha: 0.3)),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Row(
                            children: [
                              const Icon(Icons.error_outline, color: Colors.red),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Text(
                                  state.errorMessage!,
                                  style: const TextStyle(
                                      color: Colors.red, fontWeight: FontWeight.w600),
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 24),
                      ],

                      // 1. Target Role Section
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'TARGET ROLE',
                            style: GoogleFonts.inter(
                              color: AppTheme.primaryNavy,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 1.5,
                              fontSize: 13,
                            ),
                          ),
                          const SizedBox(height: 12),
                          TextFormField(
                            controller: _targetCareerController,
                            decoration: const InputDecoration(
                              hintText: 'Enter target job title (e.g. Full Stack Developer, Data Scientist)',
                              prefixIcon: Icon(Icons.work_outline, color: AppTheme.primaryNavy),
                            ),
                            validator: (val) =>
                                val == null || val.trim().isEmpty ? 'Please enter a target role' : null,
                          ),
                          const SizedBox(height: 12),
                          Wrap(
                            spacing: 8,
                            runSpacing: 8,
                            children: _suggestedRoles.map((role) {
                              final isSelected = _targetCareerController.text.trim() == role;
                              return ChoiceChip(
                                label: Text(role),
                                selected: isSelected,
                                selectedColor: AppTheme.softBlue.withValues(alpha: 0.2),
                                onSelected: (sel) {
                                  setState(() {
                                    _targetCareerController.text = sel ? role : '';
                                  });
                                },
                              );
                            }).toList(),
                          ),
                        ],
                      ).animate().fade(delay: 200.ms).slideY(begin: 0.1),

                      const SizedBox(height: 36),

                      // 2. Input Mode Toggle (File Upload vs Text)
                      Row(
                        children: [
                          Expanded(
                            child: SegmentedButton<InputMode>(
                              segments: const [
                                ButtonSegment(
                                  value: InputMode.fileUpload,
                                  label: Text('Upload PDF / Image'),
                                  icon: Icon(Icons.upload_file),
                                ),
                                ButtonSegment(
                                  value: InputMode.pasteText,
                                  label: Text('Paste Resume Text'),
                                  icon: Icon(Icons.article_outlined),
                                ),
                              ],
                              selected: {_inputMode},
                              onSelectionChanged: (set) {
                                setState(() {
                                  _inputMode = set.first;
                                });
                              },
                            ),
                          ),
                        ],
                      ).animate().fade(delay: 250.ms),

                      const SizedBox(height: 24),

                      // 3. File Upload or Text Area Box
                      if (_inputMode == InputMode.fileUpload) ...[
                        InkWell(
                          onTap: _pickFile,
                          borderRadius: BorderRadius.circular(12),
                          child: Container(
                            padding: const EdgeInsets.all(32),
                            decoration: BoxDecoration(
                              color: Colors.grey[50],
                              border: Border.all(
                                color: _selectedFileBytes != null
                                    ? AppTheme.softBlue
                                    : Colors.grey[300]!,
                                width: _selectedFileBytes != null ? 2 : 1,
                              ),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: _selectedFileBytes == null
                                ? Column(
                                    children: [
                                      const Icon(
                                        Icons.cloud_upload_outlined,
                                        size: 56,
                                        color: AppTheme.primaryNavy,
                                      ),
                                      const SizedBox(height: 16),
                                      Text(
                                        'Click to Select PDF or Image Resume',
                                        style: Theme.of(context)
                                            .textTheme
                                            .titleMedium
                                            ?.copyWith(fontWeight: FontWeight.bold),
                                      ),
                                      const SizedBox(height: 8),
                                      Text(
                                        'Supports .PDF, .PNG, .JPG, .JPEG, .WEBP files',
                                        style: TextStyle(color: Colors.grey[600], fontSize: 13),
                                      ),
                                    ],
                                  )
                                : Row(
                                    children: [
                                      Container(
                                        padding: const EdgeInsets.all(16),
                                        decoration: BoxDecoration(
                                          color: AppTheme.softBlue.withValues(alpha: 0.1),
                                          borderRadius: BorderRadius.circular(8),
                                        ),
                                        child: Icon(
                                          _selectedMimeType == 'application/pdf'
                                              ? Icons.picture_as_pdf
                                              : Icons.image,
                                          size: 36,
                                          color: AppTheme.primaryNavy,
                                        ),
                                      ),
                                      const SizedBox(width: 16),
                                      Expanded(
                                        child: Column(
                                          crossAxisAlignment: CrossAxisAlignment.start,
                                          children: [
                                            Text(
                                              _selectedFileName ?? 'Uploaded Document',
                                              style: const TextStyle(
                                                fontWeight: FontWeight.bold,
                                                fontSize: 16,
                                              ),
                                              maxLines: 1,
                                              overflow: TextOverflow.ellipsis,
                                            ),
                                            const SizedBox(height: 4),
                                            Text(
                                              '${(_selectedFileBytes!.length / 1024).toStringAsFixed(1)} KB • ${_selectedMimeType ?? "Document"}',
                                              style: TextStyle(
                                                color: Colors.grey[600],
                                                fontSize: 13,
                                              ),
                                            ),
                                          ],
                                        ),
                                      ),
                                      IconButton(
                                        onPressed: _clearSelectedFile,
                                        icon: const Icon(Icons.delete_outline, color: Colors.red),
                                        tooltip: 'Remove file',
                                      ),
                                    ],
                                  ),
                          ),
                        ).animate().fade(delay: 300.ms),
                      ] else ...[
                        TextFormField(
                          controller: _resumeController,
                          decoration: const InputDecoration(
                            hintText: 'Paste complete resume text here...',
                            alignLabelWithHint: true,
                          ),
                          maxLines: 12,
                        ).animate().fade(delay: 300.ms),
                      ],

                      const SizedBox(height: 24),

                      // 4. Optional Gemini API Key toggle
                      ExpansionTile(
                        shape: const Border(),
                        initiallyExpanded: _showGeminiKeyInput,
                        leading: const Icon(Icons.key, color: AppTheme.primaryNavy),
                        title: const Text(
                          'Gemini AI API Key (Optional)',
                          style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
                        ),
                        subtitle: const Text(
                          'Use custom Google Gemini API Key for instant Multimodal Vision analysis',
                          style: TextStyle(fontSize: 12, color: Colors.grey),
                        ),
                        children: [
                          Padding(
                            padding: const EdgeInsets.only(top: 8.0, bottom: 16.0),
                            child: TextFormField(
                              controller: _geminiKeyController,
                              obscureText: true,
                              decoration: const InputDecoration(
                                hintText: 'AIzaSy...',
                                prefixIcon: Icon(Icons.vpn_key_outlined),
                                helperText:
                                    'If empty, the backend default Gemini API Key or Local Engine will be used.',
                              ),
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 36),

                      // 5. Action Buttons (Submit, Sample, Reset)
                      Row(
                        children: [
                          Expanded(
                            child: SizedBox(
                              height: 52,
                              child: FilledButton.icon(
                                onPressed: state.isLoading ? null : _analyzeResume,
                                icon: state.isLoading
                                    ? const SizedBox(
                                        width: 20,
                                        height: 20,
                                        child: CircularProgressIndicator(
                                          color: Colors.white,
                                          strokeWidth: 2,
                                        ),
                                      )
                                    : const Icon(Icons.analytics_outlined),
                                label: Text(
                                  state.isLoading ? 'ANALYZING...' : 'SCAN & ANALYZE RESUME',
                                  style: const TextStyle(fontWeight: FontWeight.bold, letterSpacing: 1),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ).animate().fade(delay: 400.ms),

                      const SizedBox(height: 16),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          TextButton.icon(
                            onPressed: _loadSampleData,
                            icon: const Icon(Icons.lightbulb_outline, size: 18),
                            label: const Text('Try Sample Resume'),
                          ),
                          const SizedBox(width: 16),
                          TextButton.icon(
                            onPressed: _resetForm,
                            icon: const Icon(Icons.clear, size: 18),
                            label: const Text('Clear Form'),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
        ),
      ),
    );
  }

  Widget _buildResultView(BuildContext context, ResumeAnalysisResult analysis) {
    final atsScore = analysis.atsScore;
    final isGemini = analysis.analyzedBy.contains('Gemini');

    return ListView(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Scan Complete.', style: Theme.of(context).textTheme.displayMedium)
                    .animate()
                    .fade()
                    .slideY(begin: 0.1),
                const SizedBox(height: 4),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: isGemini
                        ? Colors.purple.withValues(alpha: 0.1)
                        : Colors.blue.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: isGemini ? Colors.purple : Colors.blue,
                    ),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        isGemini ? Icons.auto_awesome : Icons.memory,
                        size: 16,
                        color: isGemini ? Colors.purple : Colors.blue,
                      ),
                      const SizedBox(width: 6),
                      Text(
                        'Engine: ${analysis.analyzedBy}',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: isGemini ? Colors.purple[900] : Colors.blue[900],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            OutlinedButton.icon(
              onPressed: () => ref.read(resumeProvider.notifier).reset(),
              icon: const Icon(Icons.refresh, color: AppTheme.primaryNavy),
              label: const Text('NEW SCAN'),
            ),
          ],
        ),
        const SizedBox(height: 36),

        Container(
          color: Colors.white,
          padding: const EdgeInsets.all(36),
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
                          'OVERALL COMPATIBILITY MATCH',
                          style: GoogleFonts.inter(
                            color: Colors.grey[600],
                            fontWeight: FontWeight.w700,
                            letterSpacing: 1.5,
                            fontSize: 12,
                          ),
                        ),
                        const SizedBox(height: 12),
                        Text(
                          analysis.recommendation,
                          style: Theme.of(context)
                              .textTheme
                              .headlineMedium
                              ?.copyWith(height: 1.3, fontSize: 22),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 32),
                  Container(
                    padding: const EdgeInsets.all(28),
                    decoration: BoxDecoration(
                      border: Border.all(
                        color: atsScore >= 80
                            ? Colors.green
                            : (atsScore >= 50 ? AppTheme.burntSienna : Colors.red),
                        width: 4,
                      ),
                      shape: BoxShape.circle,
                    ),
                    child: Text(
                      '${atsScore.toInt()}%',
                      style: GoogleFonts.playfairDisplay(
                        fontSize: 42,
                        fontWeight: FontWeight.bold,
                        color: AppTheme.primaryNavy,
                      ),
                    ),
                  )
                ],
              ),
              const SizedBox(height: 36),

              if (analysis.overallAnalysis.isNotEmpty) ...[
                const Divider(),
                const SizedBox(height: 24),
                Text(
                  'DETAILED ANALYSIS',
                  style: GoogleFonts.inter(
                    color: AppTheme.primaryNavy,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1.5,
                    fontSize: 12,
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  analysis.overallAnalysis,
                  style: const TextStyle(fontSize: 15, height: 1.6, color: Colors.black87),
                ),
                const SizedBox(height: 36),
              ],

              LayoutBuilder(builder: (context, constraints) {
                final isWide = constraints.maxWidth > 500;
                final verified = Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'FOUND / DEMONSTRATED SKILLS',
                      style: GoogleFonts.inter(
                        color: Colors.green[800],
                        fontWeight: FontWeight.w700,
                        letterSpacing: 1.5,
                        fontSize: 12,
                      ),
                    ),
                    const SizedBox(height: 16),
                    analysis.skillsFound.isEmpty
                        ? const Text('No explicit skills matched.',
                            style: TextStyle(color: Colors.grey))
                        : Wrap(
                            spacing: 8,
                            runSpacing: 8,
                            children: analysis.skillsFound
                                .map((s) => Container(
                                      padding: const EdgeInsets.symmetric(
                                          horizontal: 14, vertical: 8),
                                      decoration: BoxDecoration(
                                        color: Colors.green.withValues(alpha: 0.1),
                                        borderRadius: BorderRadius.circular(6),
                                      ),
                                      child: Row(
                                        mainAxisSize: MainAxisSize.min,
                                        children: [
                                          const Icon(Icons.check_circle,
                                              size: 14, color: Colors.green),
                                          const SizedBox(width: 6),
                                          Text(
                                            s,
                                            style: const TextStyle(
                                              color: Colors.green,
                                              fontWeight: FontWeight.w600,
                                            ),
                                          ),
                                        ],
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
                      'MISSING / RECOMMENDED SKILLS',
                      style: GoogleFonts.inter(
                        color: AppTheme.burntSienna,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 1.5,
                        fontSize: 12,
                      ),
                    ),
                    const SizedBox(height: 16),
                    analysis.skillsMissing.isEmpty
                        ? const Text('None missing! High skill match.',
                            style: TextStyle(color: Colors.grey))
                        : Wrap(
                            spacing: 8,
                            runSpacing: 8,
                            children: analysis.skillsMissing
                                .map((s) => Container(
                                      padding: const EdgeInsets.symmetric(
                                          horizontal: 14, vertical: 8),
                                      decoration: BoxDecoration(
                                        color: AppTheme.burntSienna.withValues(alpha: 0.1),
                                        borderRadius: BorderRadius.circular(6),
                                      ),
                                      child: Row(
                                        mainAxisSize: MainAxisSize.min,
                                        children: [
                                          const Icon(Icons.add_circle_outline,
                                              size: 14, color: AppTheme.burntSienna),
                                          const SizedBox(width: 6),
                                          Text(
                                            s,
                                            style: const TextStyle(
                                              color: AppTheme.burntSienna,
                                              fontWeight: FontWeight.w600,
                                            ),
                                          ),
                                        ],
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
                      const SizedBox(width: 36),
                      Expanded(child: missing),
                    ],
                  );
                } else {
                  return Column(
                    children: [verified, const SizedBox(height: 36), missing],
                  );
                }
              }),

              if (analysis.topKeywords.isNotEmpty) ...[
                const SizedBox(height: 36),
                Text(
                  'DETECTED RESUME KEYWORDS',
                  style: GoogleFonts.inter(
                    color: Colors.grey[600],
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1.5,
                    fontSize: 12,
                  ),
                ),
                const SizedBox(height: 12),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: analysis.topKeywords
                      .map((kw) => Chip(
                            label: Text(kw, style: const TextStyle(fontSize: 12)),
                            backgroundColor: Colors.grey[100],
                          ))
                      .toList(),
                ),
              ],

              if (analysis.interviewQuestions.isNotEmpty) ...[
                const SizedBox(height: 48),
                const Divider(),
                const SizedBox(height: 36),
                Text(
                  'INTERVIEW PREPARATION & QUESTIONS',
                  style: GoogleFonts.inter(
                    color: AppTheme.primaryNavy,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1.5,
                    fontSize: 12,
                  ),
                ),
                const SizedBox(height: 16),
                for (var q in analysis.interviewQuestions)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 12.0),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Padding(
                          padding: EdgeInsets.only(top: 6),
                          child: Icon(Icons.help_outline, size: 18, color: AppTheme.primaryNavy),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            q,
                            style: const TextStyle(fontSize: 15, height: 1.5),
                          ),
                        ),
                      ],
                    ),
                  ),

                if (analysis.roadmapUrl != null) ...[
                  const SizedBox(height: 36),
                  Center(
                    child: OutlinedButton.icon(
                      onPressed: () async {
                        final url = Uri.parse(analysis.roadmapUrl!);
                        if (await canLaunchUrl(url)) await launchUrl(url);
                      },
                      icon: const Icon(Icons.map, color: AppTheme.primaryNavy),
                      label: const Text(
                        'EXPLORE CAREER ROADMAP',
                        style: TextStyle(
                          color: AppTheme.primaryNavy,
                          letterSpacing: 1,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 28, vertical: 20),
                        side: const BorderSide(color: AppTheme.primaryNavy, width: 2),
                      ),
                    ),
                  ),
                ]
              ]
            ],
          ),
        ).animate().fade(delay: 200.ms).slideY(begin: 0.1),

        const SizedBox(height: 36),
        Center(
          child: FilledButton.icon(
            onPressed: () => ref.read(resumeProvider.notifier).reset(),
            icon: const Icon(Icons.refresh),
            label: const Text('SCAN ANOTHER RESUME'),
          ),
        ).animate().fade(delay: 300.ms),
      ],
    );
  }
}
