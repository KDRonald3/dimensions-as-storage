"""Makes the TrueType copies (.ttf) of the Plex Sans and Plex Mono files in this folder.

    .venv/bin/python v2/writeup/fonts/convert.py

Run once; the .ttf files are kept here, so the figure script does not need this again.

Why: the typeface files from the LaTeX collection are in the OpenType format with cubic curves
(.otf). The plotting library can put such a file into a PDF only as bare outlines, which many
viewers draw unevenly at screen size. From a TrueType file it stores a proper font, and the
figures look as sharp as the text. The letter shapes are the same; only the way the curves are
written differs. LaTeX reads the .otf files, the figure script the .ttf files.

The method is the one of the fontTools project (its otf2ttf snippet).
"""

import sys

sys.dont_write_bytecode = True  # leave no cache files behind in the Python environment

from pathlib import Path

from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable

HERE = Path(__file__).resolve().parent
CLOSENESS = 1.0  # how far a rewritten curve may lie from the original, in units of 1/1000 of the type size


def truetype_copy(source, target):
    font = TTFont(source)
    names = font.getGlyphOrder()
    shapes = font.getGlyphSet()
    glyphs = {}
    for name in names:  # redraw every letter with the curves TrueType uses
        pen = TTGlyphPen(shapes)
        shapes[name].draw(Cu2QuPen(pen, CLOSENESS, reverse_direction=True))
        glyphs[name] = pen.glyph()

    # Replace the table of cubic outlines by the TrueType tables.
    font["loca"] = newTable("loca")
    font["glyf"] = table = newTable("glyf")
    table.glyphOrder = names
    table.glyphs = glyphs
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    table.compile(font)

    font["maxp"] = limits = newTable("maxp")
    limits.tableVersion = 0x00010000
    limits.maxZones = 1
    limits.maxTwilightPoints = limits.maxStorage = limits.maxFunctionDefs = 0
    limits.maxInstructionDefs = limits.maxStackElements = limits.maxSizeOfInstructions = 0
    limits.maxComponentElements = 0
    limits.compile(font)

    post = font["post"]
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = names
    font.sfntVersion = "\000\001\000\000"  # marks the file as TrueType
    font.save(target)


if __name__ == "__main__":
    for source in sorted(HERE.glob("IBMPlexSans-*.otf")) + sorted(HERE.glob("IBMPlexMono-*.otf")):
        truetype_copy(source, source.with_suffix(".ttf"))
        print("made", source.with_suffix(".ttf").name)
