/// Default configuration for Supabase integration.
/// Replace [supabaseUrl] and [supabaseAnonKey] with your project credentials
/// from your Supabase Dashboard -> Project Settings -> API.
class SupabaseConfig {
  static const String supabaseUrl = String.fromEnvironment(
    'SUPABASE_URL',
    defaultValue: 'https://your-project-id.supabase.co',
  );

  static const String supabaseAnonKey = String.fromEnvironment(
    'SUPABASE_ANON_KEY',
    defaultValue: 'your-supabase-anon-key',
  );
}
