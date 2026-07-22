import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:firebase_auth/firebase_auth.dart' as fb;
import '../../domain/models/user_model.dart';

class AuthRepository {
  /// Stream of user authentication state changes
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
      return fb.FirebaseAuth.instance.authStateChanges().map((user) {
        if (user == null) return null;
        return AppUser.fromSupabaseOrFirebase(
          id: user.uid,
          email: user.email,
          name: user.displayName,
        );
      });
    }
  }

  /// Current logged-in user
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

    try {
      final fbUser = fb.FirebaseAuth.instance.currentUser;
      if (fbUser != null) {
        return AppUser.fromSupabaseOrFirebase(
          id: fbUser.uid,
          email: fbUser.email,
          name: fbUser.displayName,
        );
      }
    } catch (_) {}

    return null;
  }

  /// Sign in user
  Future<void> signIn({required String email, required String password}) async {
    try {
      await Supabase.instance.client.auth.signInWithPassword(
        email: email,
        password: password,
      );
    } catch (sbErr) {
      await fb.FirebaseAuth.instance.signInWithEmailAndPassword(
        email: email,
        password: password,
      );
    }
  }

  /// Sign up user
  Future<void> signUp({required String email, required String password}) async {
    try {
      await Supabase.instance.client.auth.signUp(
        email: email,
        password: password,
      );
    } catch (sbErr) {
      await fb.FirebaseAuth.instance.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );
    }
  }

  /// Sign out user from both Supabase and Firebase
  Future<void> signOut() async {
    try {
      await Supabase.instance.client.auth.signOut();
    } catch (_) {}
    try {
      await fb.FirebaseAuth.instance.signOut();
    } catch (_) {}
  }
}
