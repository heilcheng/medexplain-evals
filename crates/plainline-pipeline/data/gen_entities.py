"""Regenerate entities.tsv from the Python standard library tables behind html.unescape.

The Rust cleaner must decode text exactly as `html.unescape` did in the Python version,
including legacy entities written without a semicolon, so it ships the same tables.
Usage: python gen_entities.py > entities.tsv
"""
import html
import html.entities as e

out = []
for name, value in sorted(e.html5.items()):
    out.append("E\t" + name + "\t" + " ".join(f"{ord(c):x}" for c in value))
for cp, value in sorted(html._invalid_charrefs.items()):
    out.append(f"R\t{cp:x}\t" + f"{ord(value):x}")
for cp in sorted(html._invalid_codepoints):
    out.append(f"X\t{cp:x}")
print("\n".join(out))
