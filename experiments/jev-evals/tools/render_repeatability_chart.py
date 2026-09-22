"""Render the blog's repeatability chart from saved analysis, without API calls."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/repeatability/repeat-analysis.json"
OUT = ROOT / "docs/images/repeatability-stable-labels.png"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")

WHITE = "#FFFFFF"
INK = "#26150D"
MUTED = "#63564E"
BROWN = "#3A1E10"
CREAM = "#F8F1EA"
BLUE = "#D7E9F8"
BLUE_INK = "#315D7A"
LINE = "#E4DAD1"


def font(size, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def main():
    data = json.loads(DATA.read_text())
    assert data["spans"] == 31 and data["observations_per_span_per_model"] == 3
    rows = [
        ("Jev", "jev"),
        ("Gemini 3.5 Flash-Lite", "gemini"),
        ("GPT-5.4 nano", "openai"),
    ]
    for _, key in rows:
        p = data["per_model"][key]
        assert p["stable_all_three"] + p["changed_all_three"] == 31

    im = Image.new("RGB", (1600, 900), WHITE)
    d = ImageDraw.Draw(im)
    d.text((88, 66), "Did the risk rating stay the same?", font=font(60, True), fill=INK)
    d.text((90, 147), "Same stored Bash command and rubric, rated three times by each model", font=font(29), fill=MUTED)

    d.rounded_rectangle((88, 232, 117, 261), radius=5, fill=BROWN)
    d.text((132, 231), "same label on all three calls", font=font(25), fill=INK)
    d.rounded_rectangle((558, 232, 587, 261), radius=5, fill=BLUE)
    d.text((602, 231), "label changed at least once", font=font(25), fill=INK)

    bar_x, bar_end, bar_h = 530, 1330, 60
    for i, (name, key) in enumerate(rows):
        y = 331 + i * 139
        p = data["per_model"][key]
        stable = p["stable_all_three"]
        d.text((90, y + 7), name, font=font(32, True), fill=INK)
        mask = Image.new("L", (bar_end - bar_x, bar_h), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle((0, 0, bar_end - bar_x - 1, bar_h - 1), radius=18, fill=255)
        bar = Image.new("RGB", (bar_end - bar_x, bar_h), BLUE)
        bd = ImageDraw.Draw(bar)
        bd.rectangle((0, 0, round((bar_end - bar_x) * stable / 31), bar_h), fill=BROWN)
        im.paste(bar, (bar_x, y), mask)
        d.text((1368, y + 1), f"{stable}/31", font=font(43, True), fill=INK)
        d.line((90, y + 90, 1510, y + 90), fill=LINE, width=2)

    p = data["jev_probabilities_overall"]
    assert data["per_model"]["jev"]["changed_all_three"] == 0
    d.rounded_rectangle((88, 744, 1512, 856), radius=20, fill=CREAM)
    d.rounded_rectangle((88, 744, 101, 856), radius=6, fill=BLUE_INK)
    d.text((133, 759), "Jev probability movement", font=font(29, True), fill=INK)
    d.text((133, 805),
           f"Median spread {p['chosen_prob_spread_median']:.2f}   ·   Largest spread {p['chosen_prob_spread_max']:.2f}   ·   No Jev label changes",
           font=font(27), fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    im.save(OUT, format="PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
