import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../data/repositories/resume_repository.dart';
import '../../domain/models/resume_model.dart';

final resumeRepositoryProvider = Provider<ResumeRepository>((ref) {
  return ResumeRepository();
});

class ResumeState {
  final bool isLoading;
  final String? errorMessage;
  final ResumeAnalysisResult? result;

  const ResumeState({
    this.isLoading = false,
    this.errorMessage,
    this.result,
  });

  ResumeState copyWith({
    bool? isLoading,
    String? errorMessage,
    ResumeAnalysisResult? result,
    bool clearError = false,
    bool clearResult = false,
  }) {
    return ResumeState(
      isLoading: isLoading ?? this.isLoading,
      errorMessage: clearError ? null : (errorMessage ?? this.errorMessage),
      result: clearResult ? null : (result ?? this.result),
    );
  }
}

class ResumeNotifier extends Notifier<ResumeState> {
  @override
  ResumeState build() {
    return const ResumeState();
  }

  Future<bool> analyzeResume({
    String? resumeText,
    required String targetCareer,
    List<int>? fileBytes,
    String? fileName,
    String? mimeType,
    String? geminiApiKey,
  }) async {
    final repository = ref.read(resumeRepositoryProvider);
    state = state.copyWith(isLoading: true, clearError: true);
    try {
      final res = await repository.analyzeResume(
        resumeText: resumeText,
        targetCareer: targetCareer,
        fileBytes: fileBytes,
        fileName: fileName,
        mimeType: mimeType,
        geminiApiKey: geminiApiKey,
      );

      if (res != null) {
        state = state.copyWith(isLoading: false, result: res);
        return true;
      } else {
        state = state.copyWith(
          isLoading: false,
          errorMessage: 'Resume analysis failed. Please try again.',
        );
        return false;
      }
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: 'Connection error during scan: $e',
      );
      return false;
    }
  }

  void reset() {
    state = const ResumeState();
  }
}

final resumeProvider =
    NotifierProvider<ResumeNotifier, ResumeState>(ResumeNotifier.new);
