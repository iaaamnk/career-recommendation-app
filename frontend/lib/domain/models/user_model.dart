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
    final metaName = (metadata?['name'] as String?) ?? (metadata?['full_name'] as String?);
    String displayName = 'User';

    if (metaName != null && metaName.trim().isNotEmpty) {
      displayName = metaName.trim();
    } else if (name != null && name.trim().isNotEmpty && name != 'Candidate' && name != 'User') {
      displayName = name.trim();
    } else if (email != null && email.contains('@')) {
      final username = email.split('@')[0];
      final parts = username.split(RegExp(r'[._\-]'));
      displayName = parts
          .where((p) => p.isNotEmpty)
          .map((p) => '${p[0].toUpperCase()}${p.substring(1)}')
          .join(' ');
    }

    return AppUser(
      id: id ?? '',
      email: email ?? '',
      name: displayName.isNotEmpty ? displayName : 'User',
    );
  }
}
