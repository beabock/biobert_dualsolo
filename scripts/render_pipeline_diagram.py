from PIL import Image, ImageDraw, ImageFont
import os

# Paths
md_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'pipeline_diagram.md')
out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
out_path = os.path.join(out_dir, 'pipeline_diagram.png')

# Read markdown and extract first code block
with open(md_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Find triple-backtick blocks
blocks = []
parts = text.split('```')
for i in range(1, len(parts), 2):
    blocks.append(parts[i])

if not blocks:
    raise SystemExit('No code block found in pipeline_diagram.md')

ascii_block = blocks[0].strip('\n')
lines = ascii_block.splitlines()

# Choose font
try:
    # try common monospaced font
    font = ImageFont.truetype('DejaVuSansMono.ttf', 14)
except Exception:
    font = ImageFont.load_default()

# Calculate image size
max_width_chars = max(len(l) for l in lines)
char_width, char_height = font.getsize('M')
img_width = max(800, char_width * (max_width_chars + 4))
img_height = max(600, char_height * (len(lines) + 6))

# Create image
img = Image.new('RGB', (img_width, img_height), color='white')
d = ImageDraw.Draw(img)

# Draw text with padding
padding = 10
x = padding
y = padding
for line in lines:
    d.text((x, y), line, font=font, fill='black')
    y += char_height

# Optionally add a caption footer
caption = 'Figure: Pipeline diagram (source: assets/pipeline_diagram.md)'
cap_w, cap_h = d.textsize(caption, font=font)
d.text(((img_width - cap_w) / 2, img_height - cap_h - padding), caption, font=font, fill='black')

# Ensure output dir exists
os.makedirs(out_dir, exist_ok=True)
img.save(out_path)
print('Saved', out_path)
