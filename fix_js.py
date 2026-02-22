
import os

file_path = r'c:\Users\hp\Desktop\Career\static\js\script.js'

with open(file_path, 'rb') as f:
    content = f.read()

# Remove null bytes
cleaned = content.replace(b'\x00', b'')

with open(file_path, 'wb') as f:
    f.write(cleaned)

print("JS File cleaned.")
