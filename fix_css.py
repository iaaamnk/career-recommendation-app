
import os

file_path = r'c:\Users\hp\Desktop\Career\static\css\style.css'

with open(file_path, 'rb') as f:
    content = f.read()

# Try to decode as utf-16 if it looks like it, or just clean it.
# The issue is likely mixed encoding: UTF-8 for top, UTF-16 for bottom.
# We can just read the whole thing as binary, find the spot where it goes bad, or just filter out null bytes if it's just efficient ASCII with nulls.

# Heuristic: Remove all null bytes (0x00) which are common in UTF-16le for ASCII chars.
# This might break real UTF-16 chars but for CSS it's mostly ASCII.
cleaned = content.replace(b'\x00', b'')

with open(file_path, 'wb') as f:
    f.write(cleaned)

print("File cleaned.")
