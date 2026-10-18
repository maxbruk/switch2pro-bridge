"""Minimalist high-contrast S2PC icon generator.

Outputs crisp, pixel-sharp circular icon using Switch 2 palette:
- Dark obsidian background (#12151C)
- High-contrast platinum silver ring and bold Segoe UI Black 'S2PC' text (#DCE2EE)
- Multi-DPI layers (256 down to 16) with Lanczos anti-aliasing
"""

import os
from PIL import Image, ImageDraw, ImageFont


def create_minimalist_s2pc_icon():
    size = 1024
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # High-contrast Switch 2 palette
    BG_COLOR = (18, 21, 28, 255)         # Matte obsidian/black
    BORDER_COLOR = (220, 226, 238, 255)  # Crisp platinum silver (ultra-sharp contrast)
    TEXT_COLOR = (220, 226, 238, 255)

    # Circle filling 95% of the frame (48px margin on 1024)
    padding = 28
    bbox = [padding, padding, size - padding, size - padding]
    border_width = 56

    draw.ellipse(bbox, fill=BG_COLOR, outline=BORDER_COLOR, width=border_width)

    # Ultra-bold typography (Segoe UI Black or Bahnschrift for maximum stroke thickness at small DPI)
    font_path = r"C:\Windows\Fonts\seguibl.ttf"
    if not os.path.exists(font_path):
        font_path = r"C:\Windows\Fonts\arialbd.ttf"

    font_size = 255
    font = ImageFont.truetype(font_path, font_size)

    text = "S2PC"
    t_bbox = font.getbbox(text)
    t_w = t_bbox[2] - t_bbox[0]
    t_h = t_bbox[3] - t_bbox[1]

    tx = (size - t_w) // 2 - t_bbox[0]
    ty = (size - t_h) // 2 - t_bbox[1] - 4

    draw.text((tx, ty), text, font=font, fill=TEXT_COLOR)

    return img


def save_multisize_icons(master_img: Image.Image):
    sizes = [
        (256, 256),
        (128, 128),
        (64, 64),
        (48, 48),
        (32, 32),
        (24, 24),
        (16, 16),
    ]

    # Save 256x256 preview PNG
    preview = master_img.resize((256, 256), Image.Resampling.LANCZOS)
    preview.save("app_icon.png", format="PNG")
    print("Saved preview: app_icon.png")

    artifacts_dir = r"C:\Users\brukm\.gemini\antigravity\brain\c07e988e-e6ba-4ba6-bd29-053032e54b18"
    if os.path.exists(artifacts_dir):
        preview.save(os.path.join(artifacts_dir, "app_icon.png"), format="PNG")

    # Save Windows multi-layer .ico
    master_img.save("app_icon.ico", format="ICO", sizes=sizes)
    print(f"Saved app_icon.ico with {len(sizes)} layers: {[s[0] for s in sizes]}")


if __name__ == "__main__":
    img = create_minimalist_s2pc_icon()
    save_multisize_icons(img)
