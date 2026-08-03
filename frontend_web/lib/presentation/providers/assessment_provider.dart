import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../data/repositories/career_repository.dart';
import '../../domain/models/assessment_model.dart';

final careerRepositoryProvider = Provider<CareerRepository>((ref) {
  return CareerRepository();
});

class AssessmentState {
  final bool isLoading;
  final String? errorMessage;
  final AssessmentResult? result;

  const AssessmentState({
    this.isLoading = false,
    this.errorMessage,
    this.result,
  });

  AssessmentState copyWith({
    bool? isLoading,
    String? errorMessage,
    AssessmentResult? result,
    bool clearError = false,
    bool clearResult = false,
  }) {
    return AssessmentState(
      isLoading: isLoading ?? this.isLoading,
      errorMessage: clearError ? null : (errorMessage ?? this.errorMessage),
      result: clearResult ? null : (result ?? this.result),
    );
  }
}

class AssessmentNotifier extends Notifier<AssessmentState> {
  @override
  AssessmentState build() {
    return const AssessmentState();
  }

  Future<bool> submitAssessment(AssessmentInput input) async {
    final repository = ref.read(careerRepositoryProvider);
    state = state.copyWith(isLoading: true, clearError: true);
    try {
      final res = await repository.submitAssessment(input);
      if (res != null) {
        state = state.copyWith(isLoading: false, result: res);
        return true;
      } else {
        state = state.copyWith(
          isLoading: false,
          errorMessage: 'Failed to process assessment. Please check inputs.',
        );
        return false;
      }
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: 'Error connecting to service: $e',
      );
      return false;
    }
  }

  void reset() {
    state = const AssessmentState();
  }
}

final assessmentProvider =
    NotifierProvider<AssessmentNotifier, AssessmentState>(AssessmentNotifier.new);
