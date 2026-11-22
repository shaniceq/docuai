"""Paper sizes shared by the live preview and the PDF / Word exports (points, 72 per inch).
Documents saved before paper sizes existed stay Letter."""
import math

PAPER_SIZES = {
    "letter": (612.0, 792.0),            # Letter / short bond, 8.5 x 11 in
    "a4": (595.2756, 841.8898),          # 210 x 297 mm
    "long": (612.0, 936.0),              # long bond / folio, 8.5 x 13 in
    "legal": (612.0, 1008.0),            # 8.5 x 14 in
    "a5": (419.5276, 595.2756),          # 148 x 210 mm
}


def _num(value, default):
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def paper_size(info=None, design=None):
    """(width, height) in points. The design's own stored size wins, then info["paperSize"]."""
    if isinstance(design, dict) and design.get("pw") and design.get("ph"):
        w, h = _num(design.get("pw"), 612.0), _num(design.get("ph"), 792.0)
        if 250 <= w <= 1300 and 250 <= h <= 1300:
            return (w, h)
    key = str((info or {}).get("paperSize") or "letter").lower()
    return PAPER_SIZES.get(key, PAPER_SIZES["letter"])
