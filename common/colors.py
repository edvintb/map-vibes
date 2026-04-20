"""Shared color constants and utilities."""

BG_COLOR = "#faf8f4"  # Light paper background


def darken_hex(hex_color: str, factor: float = 0.7) -> str:
    """Darken a hex color by multiplying RGB channels. Returns hex string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"


def darken_rgb(hex_color: str, factor: float = 0.7) -> tuple:
    """Darken a hex color by multiplying RGB channels. Returns (r, g, b) floats."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (r * factor, g * factor, b * factor)
