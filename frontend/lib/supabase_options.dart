/// Supabase configuration with active project credentials.
class SupabaseConfig {
  static const String supabaseUrl = String.fromEnvironment(
    'SUPABASE_URL',
    defaultValue: 'https://ecrnjlorescgpvvurqin.supabase.co',
  );

  static const String supabaseAnonKey = String.fromEnvironment(
    'SUPABASE_ANON_KEY',
    defaultValue: 'sb_publishable_oz6i5X1oSO8xkx37qiZyDQ_WHUtFYam',
  );
}
