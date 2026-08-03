import 'dart:async';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:firebase_auth/firebase_auth.dart' as fb;
import '../../domain/models/user_model.dart';

class AuthRepository {
  static AppUser? _demoUser = AppUser(
    id: 'demo-user-123',
    email: 'demo@pathfinder.ai',
    name: 'Demo Candidate',
  );

  final StreamController<AppUser?> _controller = StreamController<AppUser?>.broadcast();

  AuthRepository() {
    _controller.add(_demoUser);
  }

  /// Stream of user authentication state changes
  Stream<AppUser?> get authStateChanges {
    try {
      return Supabase.instance.client.auth.onAuthStateChange.map((data) {
        final user = data.session?.user;
        if (user == null) return _demoUser;
        return AppUser.fromSupabaseOrFirebase(
          id: user.id,
          email: user.email,
          metadata: user.userMetadata,
        );
      });
    } catch (_) {
      try {
        return fb.FirebaseAuth.instance.authStateChanges().map((user) {
          if (user == null) return _demoUser;
          return AppUser.fromSupabaseOrFirebase(
            id: user.uid,
            email: user.email,
            name: user.displayName,
          );
        });
      } catch (_) {
        return _controller.stream;
      }
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

    return _demoUser;
  }

  /// Sign in user
  Future<void> signIn({required String email, required String password}) async {
    try {
      await Supabase.instance.client.auth.signInWithPassword(
        email: email,
        password: password,
      );
    } catch (sbErr) {
      try {
        await fb.FirebaseAuth.instance.signInWithEmailAndPassword(
          email: email,
          password: password,
        );
      } catch (_) {
        _demoUser = AppUser(
          id: 'demo-${email.hashCode}',
          email: email.isNotEmpty ? email : 'guest@pathfinder.ai',
          name: email.isNotEmpty ? email.split('@')[0] : 'Guest User',
        );
        _controller.add(_demoUser);
      }
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
      try {
        await fb.FirebaseAuth.instance.createUserWithEmailAndPassword(
          email: email,
          password: password,
        );
      } catch (_) {
        _demoUser = AppUser(
          id: 'demo-${email.hashCode}',
          email: email.isNotEmpty ? email : 'guest@pathfinder.ai',
          name: email.isNotEmpty ? email.split('@')[0] : 'Guest User',
        );
        _controller.add(_demoUser);
      }
    }
  }

  /// Demo / Guest mode sign in
  Future<void> signInAsGuest() async {
    _demoUser = AppUser(
      id: 'demo-guest-user',
      email: 'guest@pathfinder.ai',
      name: 'Guest Candidate',
    );
    _controller.add(_demoUser);
  }

  /// Sign out user
  Future<void> signOut() async {
    _demoUser = null;
    _controller.add(null);
    try {
      await Supabase.instance.client.auth.signOut();
    } catch (_) {}
    try {
      await fb.FirebaseAuth.instance.signOut();
    } catch (_) {}
  }
}
