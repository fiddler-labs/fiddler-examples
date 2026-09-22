"""Render all stored repeat ratings as model clusters for the blog.

Each row is the same command across the three model panels. The three tiles in
each panel are the original rating and two repeats, read directly from the
saved repeatability artifact. This script makes no model or network calls.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/repeatability/repeat-per-span.json"
OUT = ROOT / "docs/images/repeatability-risk-paths.png"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")

WHITE = "#FFFFFF"
INK = "#26150D"
MUTED = "#63564E"
BROWN = "#3A1E10"
CREAM = "#F8F1EA"
PALE_BLUE = "#EAF3FA"
BLUE_INK = "#315D7A"
LINE = "#E4DAD1"
RISK = {
    "low": ("#5F9D77", "#FFFFFF", "L"),
    "medium": ("#D4A83B", INK, "M"),
    "high": ("#D37B43", "#FFFFFF", "H"),
    "critical": ("#B94E46", "#FFFFFF", "C"),
}
MODELS = [("Jev", "jev"), ("Gemini 3.5 Flash-Lite", "gemini"), ("GPT-5.4 nano", "openai")]
CALLS = ("baseline", "r1", "r2")


def font(size, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def center_text(draw, xy, value, text_font, fill):
    x, y = xy
    box = draw.textbbox((0, 0), value, font=text_font)
    draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - box[1]),
              value, font=text_font, fill=fill)


def draw_tile(draw, x, y, label):
    fill, text_color, initial = RISK[label]
    draw.rounded_rectangle((x, y, x + 72, y + 34), radius=7, fill=fill)
    center_text(draw, (x + 36, y + 17), initial, font(23, True), text_color)


def main():
    rows = json.loads(DATA.read_text())
    assert len(rows) == 31 and len({r["span_id"] for r in rows}) == 31
    sequences = {}
    for _, model in MODELS:
        sequences[model] = []
        for row in rows:
            labels = row["models"][model]["labels"]
            seq = tuple(labels[call] for call in CALLS)
            assert all(label in RISK for label in seq)
            sequences[model].append(seq)

    width, height = 1600, 1980
    im = Image.new("RGB", (width, height), WHITE)
    d = ImageDraw.Draw(im)
    d.text((78, 48), "How much did the ratings vary?", font=font(57, True), fill=INK)
    d.text((80, 123), "All 31 stored commands, rated three times by each model", font=font(29), fill=MUTED)
    top = 345
    row_h = 46
    model_x = [162, 626, 1090]
    panel_w = 428
    tile_step = 112
    tile_left = 66

    d.text((79, 294), "COMMAND", font=font(22, True), fill=MUTED)
    for (name, model), px in zip(MODELS, model_x):
        count = sum(len(set(seq)) > 1 for seq in sequences[model])
        d.text((px + 8, 205), name, font=font(30, True), fill=INK)
        d.text((px + 8, 250), f"{count} of 31 changed", font=font(23), fill=BLUE_INK)
        for i, call in enumerate(("ORIG", "R1", "R2")):
            center_text(d, (px + tile_left + i * tile_step + 36, 308),
                        call, font(19, True), MUTED)
        d.line((px, 325, px + panel_w, 325), fill=LINE, width=2)

    for row_index, _row in enumerate(rows):
        y = top + row_index * row_h
        if row_index % 2 == 0:
            d.rounded_rectangle((76, y - 3, 1524, y + 38), radius=5, fill=CREAM)
        d.text((83, y + 5), f"{row_index + 1:02d}", font=font(23), fill=MUTED)
        for (_, model), px in zip(MODELS, model_x):
            seq = sequences[model][row_index]
            if len(set(seq)) > 1:
                d.rounded_rectangle((px + 5, y - 2, px + panel_w - 5, y + 38),
                                    radius=8, fill=PALE_BLUE, outline=BLUE_INK, width=2)
            for i, label in enumerate(seq):
                draw_tile(d, px + tile_left + i * tile_step, y + 1, label)

    footer_y = top + len(rows) * row_h + 43
    d.line((80, footer_y - 23, 1520, footer_y - 23), fill=LINE, width=2)
    legend_y = footer_y
    x = 80
    for label in RISK:
        fill, text_color, initial = RISK[label]
        d.rounded_rectangle((x, legend_y, x + 31, legend_y + 31), radius=5, fill=fill)
        center_text(d, (x + 15.5, legend_y + 15.5), initial, font(21, True), text_color)
        d.text((x + 42, legend_y + 1), label, font=font(25), fill=INK)
        x += 190
    d.rounded_rectangle((1030, legend_y - 3, 1072, legend_y + 34), radius=8,
                        fill=PALE_BLUE, outline=BLUE_INK, width=3)
    d.text((1086, legend_y + 1), "changed at least once", font=font(25), fill=INK)
    d.text((80, footer_y + 52), "Each row is the same command across models · original → repeat 1 → repeat 2",
           font=font(25), fill=BROWN)
    d.text((80, footer_y + 94), "Consistency describes repeated labels; it does not establish accuracy.",
           font=font(25), fill=MUTED)

    assert footer_y + 130 < height
    OUT.parent.mkdir(parents=True, exist_ok=True)
    im.save(OUT, format="PNG", optimize=True)
    print(OUT)
    for _, model in MODELS:
        print(f"{model}: {sum(len(set(s)) > 1 for s in sequences[model])} changed / 31")


if __name__ == "__main__":
    main()
