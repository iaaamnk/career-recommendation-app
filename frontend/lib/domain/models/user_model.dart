class AppUser {
  final String id;
  final String email;
  final String name;

  const AppUser({
    required this.id,
    required this.email,
    required this.name,
  });

  factory AppUser.fromSupabaseOrFirebase({
    String? id,
    String? email,
    String? name,
    Map<String, dynamic>? metadata,
  }) {
    final metaName = metadata?['name'] as String?;
    final displayName = metaName ?? name ?? (email != null && email.contains('@') ? email.split('@')[0] : 'User');

    return AppUser(
      id: id ?? '',
      email: email ?? '',
      name: displayName,
    );
  }
}
