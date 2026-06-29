import os

file_path = "lib/main.dart"

with open(file_path, 'r') as f:
    content = f.read()

# Replace background
content = content.replace("0xFFF7F5F0", "0xFFF4F2EF")

# Replace dark color (charcoal) with Navy
content = content.replace("0xFF1A1A1A", "0xFF213E60")

# Replace primary accent with Orange
content = content.replace("0xFFE76F51", "0xFFE68C3A")

# Replace the Teal with the Light Blue
content = content.replace("0xFF2A9D8F", "0xFF94B6EF")

# Replace some other hardcoded colors to use the palette
# For example, let's use the light blue for the 'VERIFIED SKILLS' bg
# and maybe the Auth screen left panel can have a gradient between Navy and Light Blue?
auth_bg = """              child: Container(
                color: const Color(0xFF213E60),"""
auth_bg_new = """              child: Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Color(0xFF213E60), Color(0xFF94B6EF)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),"""
content = content.replace(auth_bg, auth_bg_new)

# In the ATS Score circle
ats_circle = """border: Border.all(color: atsScore > 75 ? const Color(0xFF94B6EF) : (atsScore > 50 ? const Color(0xFFE9C46A) : const Color(0xFFE68C3A)), width: 4),"""
ats_circle_new = """border: Border.all(color: atsScore > 75 ? const Color(0xFF94B6EF) : (atsScore > 50 ? const Color(0xFFE68C3A) : Colors.red), width: 4),"""
content = content.replace(ats_circle, ats_circle_new)

with open(file_path, 'w') as f:
    f.write(content)
