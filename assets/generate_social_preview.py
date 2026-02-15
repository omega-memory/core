"""Generate social preview image for omega-memory/omega-memory GitHub repo."""
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 640

# Colors
BG_TOP = (13, 17, 23)       # GitHub dark
BG_BOT = (22, 27, 38)       # Slightly lighter
WHITE = (255, 255, 255)
LIGHT_GRAY = (139, 148, 158)
ACCENT = (88, 101, 242)     # Blurple accent
GREEN = (63, 185, 80)       # GitHub green
DIM = (110, 118, 129)
PILL_BG = (33, 38, 52)
TERMINAL_BG = (22, 27, 34)
TERMINAL_BORDER = (48, 54, 61)

# Fonts
FONT_BOLD_72 = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 72)
FONT_BOLD_28 = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 28)
FONT_REG_24 = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 24)
FONT_REG_20 = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 20)
FONT_MONO_18 = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 18)
FONT_MONO_16 = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 16)
FONT_BOLD_20 = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 20)
FONT_BOLD_22 = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 22)

img = Image.new("RGB", (WIDTH, HEIGHT))
draw = ImageDraw.Draw(img)

# Gradient background
for y in range(HEIGHT):
    t = y / HEIGHT
    r = int(BG_TOP[0] * (1 - t) + BG_BOT[0] * t)
    g = int(BG_TOP[1] * (1 - t) + BG_BOT[1] * t)
    b = int(BG_TOP[2] * (1 - t) + BG_BOT[2] * t)
    draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))

# === LEFT SIDE (text content) ===
LEFT_X = 80

# Title
draw.text((LEFT_X, 80), "OMEGA", fill=WHITE, font=FONT_BOLD_72)

# Tagline
draw.text((LEFT_X, 175), "Persistent memory for AI coding agents", fill=LIGHT_GRAY, font=FONT_BOLD_28)

# Benchmark badge
badge_y = 240
badge_text = "#1 on LongMemEval  •  95.4%"
bbox = draw.textbbox((0, 0), badge_text, font=FONT_BOLD_20)
bw = bbox[2] - bbox[0] + 32
bh = bbox[3] - bbox[1] + 20
draw.rounded_rectangle(
    [LEFT_X, badge_y, LEFT_X + bw, badge_y + bh],
    radius=8, fill=PILL_BG, outline=ACCENT, width=2,
)
draw.text((LEFT_X + 16, badge_y + 8), badge_text, fill=ACCENT, font=FONT_BOLD_20)

# Feature pills
pill_y = badge_y + bh + 30
pills = ["Local-first", "25 MCP tools", "Semantic search", "Zero cloud"]
px = LEFT_X
for pill in pills:
    bbox = draw.textbbox((0, 0), pill, font=FONT_REG_20)
    pw = bbox[2] - bbox[0] + 24
    ph = bbox[3] - bbox[1] + 14
    draw.rounded_rectangle(
        [px, pill_y, px + pw, pill_y + ph],
        radius=6, fill=PILL_BG,
    )
    draw.text((px + 12, pill_y + 5), pill, fill=LIGHT_GRAY, font=FONT_REG_20)
    px += pw + 10

# Client badges
client_y = pill_y + 55
draw.text((LEFT_X, client_y), "Works with:", fill=DIM, font=FONT_REG_20)
clients = ["Claude Code", "Cursor", "Windsurf"]
cx = LEFT_X + 120
for client in clients:
    bbox = draw.textbbox((0, 0), client, font=FONT_REG_20)
    cw = bbox[2] - bbox[0] + 20
    ch = bbox[3] - bbox[1] + 12
    draw.rounded_rectangle(
        [cx, client_y - 2, cx + cw, client_y + ch - 2],
        radius=5, fill=(30, 35, 48),
    )
    draw.text((cx + 10, client_y + 3), client, fill=WHITE, font=FONT_REG_20)
    cx += cw + 8

# Install command
install_y = client_y + 55
draw.text((LEFT_X, install_y), "$ ", fill=GREEN, font=FONT_MONO_18)
draw.text((LEFT_X + 24, install_y), "pip install omega-memory", fill=WHITE, font=FONT_MONO_18)

# GitHub URL
draw.text((LEFT_X, HEIGHT - 60), "github.com/omega-memory/omega-memory", fill=DIM, font=FONT_REG_20)

# === RIGHT SIDE (terminal mockup) ===
TERM_X = 700
TERM_Y = 80
TERM_W = 500
TERM_H = 440

# Terminal window
draw.rounded_rectangle(
    [TERM_X, TERM_Y, TERM_X + TERM_W, TERM_Y + TERM_H],
    radius=12, fill=TERMINAL_BG, outline=TERMINAL_BORDER, width=1,
)

# Terminal title bar dots
for i, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
    draw.ellipse(
        [TERM_X + 16 + i * 22, TERM_Y + 14, TERM_X + 28 + i * 22, TERM_Y + 26],
        fill=color,
    )

# Terminal content
ty = TERM_Y + 48
line_h = 26

lines = [
    (GREEN,  "$ omega setup"),
    (GREEN,  "✓ Memory system ready"),
    (None, ""),
    (DIM,    "# 3 weeks later, new session..."),
    (None, ""),
    (WHITE,  '> "What did we decide about'),
    (WHITE,  '   the auth approach?"'),
    (None, ""),
    (ACCENT, "→ Decision (3 weeks ago):"),
    (WHITE,  '  "We chose JWT tokens over'),
    (WHITE,  '   session cookies because the'),
    (WHITE,  '   API needs to be stateless'),
    (WHITE,  '   for horizontal scaling."'),
    (None, ""),
    (DIM,    "  Relevance: 0.94 • Accessed 7×"),
]

for color, text in lines:
    if color is not None:
        draw.text((TERM_X + 20, ty), text, fill=color, font=FONT_MONO_16)
    ty += line_h

img.save("/Users/singularityjason/Projects/omega-public/assets/social-preview.png", quality=95)
print(f"Saved: 1280x640")
