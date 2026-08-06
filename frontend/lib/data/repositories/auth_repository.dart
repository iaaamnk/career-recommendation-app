import 'dart:async';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../../domain/models/user_model.dart';

class AuthRepository {
  final StreamController<AppUser?> _controller = StreamController<AppUser?>.broadcast();

  AuthRepository() {
    _controller.add(currentUser);
  }

  /// Stream of Supabase user authentication state changes
  Stream<AppUser?> get authStateChanges {
    try {
      return Supabase.instance.client.auth.onAuthStateChange.map((data) {
        final user = data.session?.user;
        if (user == null) return null;
        return AppUser.fromSupabaseOrFirebase(
          id: user.id,
          email: user.email,
          metadata: user.userMetadata,
        );
      });
    } catch (_) {
      return _controller.stream;
    }
  }

  /// Current logged-in Supabase user
  AppUser? get currentUser {
    try {
      final sbUser = Supabase.instance.client.auth.currentUser;
      if (sbUser != null) {
        return AppUser.fromSupabaseOrFirebase(
          id: sbUser.id,
          email: sbUser.email,
          metadata: sbUser.userMetadata,
        );
      }
    } catch (_) {}

    return null;
  }

  /// Sign in user with Supabase
  Future<void> signIn({required String email, required String password}) async {
    final response = await Supabase.instance.client.auth.signInWithPassword(
      email: email,
      password: password,
    );
    if (response.user != null) {
      _controller.add(AppUser.fromSupabaseOrFirebase(
        id: response.user!.id,
        email: response.user!.email,
        metadata: response.user!.userMetadata,
      ));
    }
  }

  /// Sign up user with Supabase
  Future<void> signUp({required String email, required String password}) async {
    final response = await Supabase.instance.client.auth.signUp(
      email: email,
      password: password,
    );
    if (response.user != null) {
      _controller.add(AppUser.fromSupabaseOrFirebase(
        id: response.user!.id,
        email: response.user!.email,
        metadata: response.user!.userMetadata,
      ));
    }
  }

  /// Sign out user from Supabase
  Future<void> signOut() async {
    _controller.add(null);
    try {
      await Supabase.instance.client.auth.signOut();
    } catch (_) {}
  }
}
