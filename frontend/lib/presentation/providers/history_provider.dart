import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'assessment_provider.dart';
import '../../domain/models/history_model.dart';

final historyFutureProvider = FutureProvider.autoDispose<HistoryData?>((ref) async {
  final repository = ref.watch(careerRepositoryProvider);
  return await repository.fetchHistory();
});
