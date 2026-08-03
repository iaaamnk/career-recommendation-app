import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  static const Color primaryNavy = Color(0xFF213E60);
  static const Color burntSienna = Color(0xFFE68C3A);
  static const Color softBlue = Color(0xFF94B6EF);
  static const Color backgroundColor = Color(0xFFF4F2EF);

  static ThemeData get lightTheme {
    return ThemeData(
      useMaterial3: true,
      scaffoldBackgroundColor: backgroundColor,
      colorScheme: ColorScheme.fromSeed(
        seedColor: burntSienna,
        primary: primaryNavy,
        secondary: burntSienna,
        surface: backgroundColor,
        onSurface: primaryNavy,
      ),
      textTheme: TextTheme(
        displayLarge: GoogleFonts.playfairDisplay(
          fontWeight: FontWeight.w700,
          color: primaryNavy,
          letterSpacing: -1.5,
        ),
        displayMedium: GoogleFonts.playfairDisplay(
          fontWeight: FontWeight.w700,
          color: primaryNavy,
          letterSpacing: -1.0,
        ),
        headlineMedium: GoogleFonts.playfairDisplay(
          fontWeight: FontWeight.w600,
          color: primaryNavy,
        ),
        titleLarge: GoogleFonts.inter(
          fontWeight: FontWeight.w600,
          color: primaryNavy,
          letterSpacing: -0.5,
        ),
        bodyLarge: GoogleFonts.inter(
          color: const Color(0xFF4A4A4A),
          height: 1.6,
        ),
        bodyMedium: GoogleFonts.inter(
          color: const Color(0xFF4A4A4A),
          height: 1.5,
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.white,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(0),
          borderSide: BorderSide.none,
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(0),
          borderSide: const BorderSide(color: primaryNavy, width: 2),
        ),
        contentPadding: const EdgeInsets.all(20),
        labelStyle: GoogleFonts.inter(color: Colors.grey[600]),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: primaryNavy,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 32),
          shape: const RoundedRectangleBorder(borderRadius: BorderRadius.zero),
          textStyle: GoogleFonts.inter(
            fontWeight: FontWeight.w600,
            letterSpacing: 1,
          ),
        ),
      ),
      cardTheme: const CardThemeData(
        color: Colors.white,
        elevation: 0,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.zero),
        margin: EdgeInsets.zero,
      ),
    );
  }
}
