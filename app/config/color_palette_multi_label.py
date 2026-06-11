import math
from collections import defaultdict
import numpy as np
import matplotlib.colors as mcolors

from skimage import color as skcolor
from skimage.color import deltaE_ciede2000

class ColorPaletteMultiLabel:
    TARGET_COLORS_REF = [

    # ── PUTIH ──────────────────────────────────────────────────────────────────
    ('putih',   [255, 255, 255]),  # Pure White
    ('putih',   [255, 254, 253]),  # Barely White
    ('putih',   [255, 253, 250]),  # Floral White
    ('putih',   [255, 252, 248]),  # Snow White
    ('putih',   [255, 251, 245]),  # Old Lace
    ('putih',   [255, 250, 242]),  # Seashell
    ('putih',   [255, 249, 240]),  # Cosmic Latte
    ('putih',   [254, 252, 250]),  # White Smoke Warm
    ('putih',   [253, 252, 251]),  # Ghost White Warm
    ('putih',   [252, 251, 250]),  # Off White Neutral
    ('putih',   [252, 250, 248]),  # Pearl White
    ('putih',   [251, 250, 249]),  # Magnolia
    ('putih',   [250, 250, 250]),  # White Smoke
    ('putih',   [250, 249, 248]),  # Baby Powder
    ('putih',   [249, 248, 246]),  # Soft White
    ('putih',   [248, 247, 245]),  # Isabelline
    ('putih',   [247, 246, 244]),  # Cultured
    ('putih',   [246, 245, 243]),  # Near White
    ('putih',   [245, 244, 242]),  # Light White Gray
    ('putih',   [244, 243, 241]),  # Pale Smoke
    ('putih',   [243, 242, 240]),  # Lavender Blush White
    ('putih',   [242, 241, 239]),  # Soft Ivory
    ('putih',   [241, 240, 238]),  # Alabaster
    ('putih',   [240, 239, 237]),  # Anti-Flash White
    ('putih',   [238, 237, 235]),  # Bright White
    ('putih',   [236, 235, 233]),  # White Ash
    ('putih',   [235, 234, 232]),  # Soft Ash
    ('putih',   [233, 232, 230]),  # Light Ash
    ('putih',   [231, 230, 228]),  # Pale Ash
    ('putih',   [229, 228, 226]),  # Silver White
    ('putih',   [227, 226, 224]),  # Platinum White
    ('putih',   [225, 224, 222]),  # Cloud White
    ('putih',   [223, 222, 220]),  # Pale Pewter
    ('putih',   [221, 220, 218]),  # Gainsboro Warm
    ('putih',   [219, 218, 216]),  # Gainsboro

    # ── KREM ───────────────────────────────────────────────────────────────────
    ('krem',    [255, 255, 240]),  # Ivory
    ('krem',    [255, 254, 230]),  # Pale Ivory
    ('krem',    [255, 253, 220]),  # Cornsilk
    ('krem',    [255, 252, 210]),  # Light Ivory
    ('krem',    [255, 251, 200]),  # Cream Yellow
    ('krem',    [255, 250, 190]),  # Butter Cream
    ('krem',    [255, 248, 185]),  # Warm Ivory
    ('krem',    [255, 247, 237]),  # Warm White
    ('krem',    [255, 251, 235]),  # Ivory White
    ('krem',    [253, 248, 230]),  # Parchment Light
    ('krem',    [252, 245, 220]),  # Linen Light
    ('krem',    [250, 248, 246]),  # Warm Off White
    ('krem',    [250, 245, 225]),  # Antique White Light
    ('krem',    [248, 244, 218]),  # Antique White
    ('krem',    [245, 245, 244]),  # Stone White
    ('krem',    [245, 240, 228]),  # Cream Hangat
    ('krem',    [244, 238, 220]),  # Light Beige
    ('krem',    [242, 236, 214]),  # Pale Parchment
    ('krem',    [240, 235, 220]),  # Warm Beige
    ('krem',    [238, 232, 210]),  # Parchment
    ('krem',    [235, 230, 215]),  # Dark Beige
    ('krem',    [232, 225, 205]),  # Sandy Beige
    ('krem',    [230, 225, 210]),  # Linen Beige Tua
    ('krem',    [228, 220, 198]),  # Beige
    ('krem',    [225, 218, 193]),  # Bisque Beige
    ('krem',    [220, 210, 185]),  # Dark Cream
    ('krem',    [215, 205, 178]),  # Wheat Beige
    ('krem',    [210, 200, 170]),  # Warm Tan Light
    ('krem',    [205, 193, 162]),  # Sand Beige
    ('krem',    [200, 188, 155]),  # Tan Beige
    ('krem',    [195, 182, 145]),  # Dark Sand
    ('krem',    [190, 175, 135]),  # Khaki Light
    ('krem',    [185, 168, 125]),  # Pale Khaki

    # ── HITAM ──────────────────────────────────────────────────────────────────
    ('hitam',   [0,   0,   0]),    # Pure Black
    ('hitam',   [5,   5,   5]),    # Absolute Black
    ('hitam',   [10,  10,  10]),   # Near Black
    ('hitam',   [15,  15,  15]),   # Very Dark Black
    ('hitam',   [20,  20,  20]),   # Rich Black
    ('hitam',   [25,  25,  25]),   # Dark Neutral Black
    ('hitam',   [30,  30,  30]),   # Licorice
    ('hitam',   [35,  35,  35]),   # Soft Black
    ('hitam',   [40,  40,  40]),   # Oil Black
    ('hitam',   [45,  45,  45]),   # Faded Black
    ('hitam',   [48,  46,  46]),   # Warm Faded Black
    ('hitam',   [46,  48,  50]),   # Cool Faded Black
    ('hitam',   [50,  48,  48]),   # Warm Charcoal
    ('hitam',   [48,  50,  52]),   # Cool Charcoal
    ('hitam',   [52,  50,  48]),   # Brown Tinted Black
    ('hitam',   [55,  55,  55]),   # Charcoal Black
    ('hitam',   [58,  58,  58]),   # Dark Charcoal
    ('hitam',   [60,  60,  60]),   # Medium Charcoal
    ('hitam',   [62,  60,  58]),   # Warm Medium Charcoal
    ('hitam',   [58,  60,  62]),   # Cool Medium Charcoal
    ('hitam',   [65,  63,  63]),   # Dim Gray Dark
    ('hitam',   [65,  65,  65]),   # Graphite Black
    ('hitam',   [68,  66,  64]),   # Warm Graphite
    ('hitam',   [64,  66,  68]),   # Cool Graphite

    # ── ABU-ABU ────────────────────────────────────────────────────────────────
    ('abu-abu', [75,  73,  73]),   # Dark Gray
    ('abu-abu', [78,  78,  78]),   # Dim Gray
    ('abu-abu', [80,  80,  82]),   # Cool Dim Gray
    ('abu-abu', [82,  80,  78]),   # Warm Dim Gray
    ('abu-abu', [85,  85,  85]),   # Dark Gray Netral
    ('abu-abu', [88,  86,  84]),   # Gunmetal Gray
    ('abu-abu', [90,  88,  87]),   # Warm Dark Gray
    ('abu-abu', [92,  92,  94]),   # Cool Dark Gray
    ('abu-abu', [95,  95,  95]),   # Mortar Gray
    ('abu-abu', [98,  96,  94]),   # Warm Medium Dark Gray
    ('abu-abu', [100, 100, 100]),  # Medium Dark Gray
    ('abu-abu', [103, 103, 105]),  # Cool Medium Gray
    ('abu-abu', [105, 103, 103]),  # Warm Medium Gray
    ('abu-abu', [108, 108, 108]),  # Sonic Silver
    ('abu-abu', [110, 110, 110]),  # Old Silver
    ('abu-abu', [110, 110, 112]),  # Cool Medium Gray 2
    ('abu-abu', [112, 110, 108]),  # Warm Old Silver
    ('abu-abu', [115, 115, 115]),  # Medium Gray
    ('abu-abu', [118, 116, 114]),  # Warm Gray Mid
    ('abu-abu', [120, 120, 122]),  # Spanish Gray
    ('abu-abu', [122, 122, 122]),  # Gray
    ('abu-abu', [125, 122, 120]),  # Warm Gray
    ('abu-abu', [128, 128, 130]),  # Cool Gray
    ('abu-abu', [130, 130, 130]),  # Silver Chalice
    ('abu-abu', [130, 130, 132]),  # Cool Gray 2
    ('abu-abu', [133, 131, 129]),  # Warm Silver
    ('abu-abu', [135, 135, 135]),  # Trolley Gray
    ('abu-abu', [140, 140, 140]),  # Gray Netral
    ('abu-abu', [143, 141, 139]),  # Warm Light Gray
    ('abu-abu', [145, 145, 147]),  # Cool Light Gray
    ('abu-abu', [148, 148, 148]),  # Dark Silver
    ('abu-abu', [150, 148, 146]),  # Warm Light Gray 2
    ('abu-abu', [152, 152, 154]),  # Philippne Silver
    ('abu-abu', [155, 155, 157]),  # Cool Light Gray 2
    ('abu-abu', [158, 158, 158]),  # Light Gray Mid
    ('abu-abu', [160, 160, 160]),  # Silver Sand
    ('abu-abu', [163, 163, 163]),  # Light Gray
    ('abu-abu', [166, 164, 162]),  # Warm Pale Gray Mid
    ('abu-abu', [168, 168, 170]),  # Cool Pale Gray
    ('abu-abu', [170, 170, 170]),  # Silver
    ('abu-abu', [173, 171, 169]),  # Warm Silver 2
    ('abu-abu', [175, 172, 170]),  # Warm Pale Gray
    ('abu-abu', [176, 176, 178]),  # Lavender Gray
    ('abu-abu', [178, 178, 178]),  # Ash Gray
    # ('abu-abu', [180, 180, 182]),  # Cool Pale Gray 2
    ('abu-abu', [182, 180, 178]),  # Warm Ash Gray
    ('abu-abu', [185, 185, 185]),  # Pale Silver
    # ('abu-abu', [188, 186, 184]),  # Warm Very Light Gray
    # ('abu-abu', [190, 190, 190]),  # Pale Gray
    ('abu-abu', [193, 193, 195]),  # Cool Very Light Gray
    ('abu-abu', [196, 196, 196]),  # Light Silver
    # ('abu-abu', [200, 198, 196]),  # Very Light Warm Gray
    # ('abu-abu', [203, 203, 205]),  # Platinum
    # ('abu-abu', [206, 206, 206]),  # Very Light Gray
    # ('abu-abu', [210, 210, 212]),  # Near White Cool Gray 2
    ('abu-abu', [212, 212, 212]),  # Smoke White
    # ('abu-abu', [212, 212, 216]),  # Near White Cool Gray
    ('abu-abu', [215, 215, 215]),  # Pale Gray Light
    # ('abu-abu', [218, 218, 218]),  # Gainsboro Gray
    # ('abu-abu', [220, 220, 222]),  # Near White Gray
    ('abu-abu', [222, 222, 220]),  # Warm Near White

    # ── MERAH ──────────────────────────────────────────────────────────────────
    ('merah',   [248, 200, 200]),  # Misty Rose
    ('merah',   [248, 180, 180]),  # Pale Pink Red
    ('merah',   [248, 160, 160]),  # Light Salmon Red
    ('merah',   [248, 140, 140]),  # Salmon Red
    ('merah',   [248, 113, 113]),  # Light Red
    ('merah',   [240, 100, 100]),  # Soft Red
    # ('merah',   [239, 68,  68]),   # Red
    ('merah',   [230, 60,  60]),   # Strong Red
    ('merah',   [220, 55,  55]),   # Medium Red
    ('merah',   [210, 50,  50]),   # Dark Red Medium
    ('merah',   [210, 50,  40]),   # Medium Red
    ('merah',   [200, 45,  40]),   # Venetian Red
    ('merah',   [195, 68,  28]),   # Deep Rust Orange
    ('merah',   [190, 65,  25]),   # Rust Orange Gelap
    ('merah',   [190, 70,  55]),   # Rose Red Hangat
    ('merah',   [190, 40,  35]),   # Crimson Dark
    ('merah',   [185, 62,  22]),   # Dark Rust Orange
    ('merah',   [185, 28,  28]),   # Deep Red
    ('merah',   [180, 35,  35]),   # Cardinal Red
    ('merah',   [175, 80,  65]),   # Warm Rose Red
    ('merah',   [175, 30,  30]),   # Firebrick
    ('merah',   [170, 75,  30]),   # Rust Red Sedang
    ('merah',   [170, 50,  35]),   # Brick Red Terang
    ('merah',   [165, 75,  60]),   # Merah Terracotta Terang
    ('merah',   [160, 60,  45]),   # Light Brick Red
    ('merah',   [160, 45,  32]),   # Merah Bata Cerah Sedang
    ('merah',   [155, 55,  42]),   # Brick Red Hangat
    # ('merah',   [155, 40,  16]),   # Brick Red Tua
    # ('merah',   [154, 60,  25]),   # Rust Red Gelap
    # ('merah',   [150, 40,  30]),   # Merah Bata Sedang
    # ('merah',   [150, 38,  14]),   # Merah Bata Gelap
    # ('merah',   [145, 50,  40]),   # Brick Red Sedang
    # ('merah',   [145, 35,  12]),   # Brick Red Pekat Tua
    # ('merah',   [140, 70,  70]),   # Dusty Red
    # ('merah',   [140, 35,  28]),   # Brick Red Sedang Gelap
    # ('merah',   [135, 45,  35]),   # Dark Brick Red
    ('merah',   [127, 29,  29]),   # Dark Red
    ('merah',   [120, 55,  55]),   # Dusty Red Tua
    ('merah',   [120, 32,  22]),   # Deep Brick Red
    ('merah',   [110, 30,  25]),   # Brick Red Pekat
    ('merah',   [100, 40,  35]),   # Merah Gelap Hangat
    ('merah',   [95,  22,  18]),   # Dark Crimson
    ('merah',   [85,  20,  20]),   # Blood Red
    ('merah',   [75,  18,  18]),   # Marun Gelap
    ('merah',   [60,  15,  15]),   # Marun Sangat Gelap

    # ── ORANYE ─────────────────────────────────────────────────────────────────
    ('oranye',  [255, 220, 180]),  # Peach Light
    ('oranye',  [255, 200, 150]),  # Light Peach
    ('oranye',  [255, 185, 120]),  # Pale Orange
    ('oranye',  [255, 170, 100]),  # Light Apricot
    # ('oranye',  [255, 160, 80]),   # Peach Orange
    # ('oranye',  [255, 150, 65]),   # Apricot
    # ('oranye',  [255, 140, 55]),   # Bright Orange Light
    # ('oranye',  [255, 130, 50]),   # Light Tangerine
    # ('oranye',  [255, 120, 45]),   # Bright Orange
    # ('oranye',  [255, 110, 35]),   # Tangerine
    # ('oranye',  [255, 100, 30]),   # Vivid Orange
    # ('oranye',  [252, 175, 95]),   # Sandy Orange
    # ('oranye',  [251, 160, 75]),   # Light Orange 2
    # ('oranye',  [251, 146, 60]),   # Light Orange
    # ('oranye',  [250, 135, 50]),   # Orange Soft
    # ('oranye',  [249, 115, 22]),   # Orange
    # ('oranye',  [245, 140, 70]),   # Orange Muda Hangat
    # ('oranye',  [240, 130, 60]),   # Orange Terang Sedang
    # ('oranye',  [234, 88,  12]),   # Dark Orange
    # ('oranye',  [230, 140, 70]),   # Terracotta Terang
    # ('oranye',  [225, 115, 55]),   # Deep Orange Hangat
    # ('oranye',  [220, 130, 65]),   # Terracotta Sedang
    ('oranye',  [215, 110, 50]),   # Burnt Orange
    ('oranye',  [210, 120, 70]),   # Terracotta
    ('oranye',  [210, 100, 45]),   # Deep Terracotta
    ('oranye',  [205, 95,  42]),   # Burnt Orange Sedang
    ('oranye',  [200, 100, 40]),   # Dark Burnt Orange
    ('oranye',  [195, 90,  38]),   # Burnt Sienna
    ('oranye',  [190, 90,  35]),   # Rust Orange
    ('oranye',  [185, 82,  30]),   # Dark Rust Light
    ('oranye',  [180, 83,  9]),    # Dark Amber Orange
    ('oranye',  [175, 78,  20]),   # Russet Orange
    ('oranye',  [170, 72,  18]),   # Copper Orange
    ('oranye',  [165, 68,  16]),   # Amber Orange Dark
    ('oranye',  [160, 65,  15]),   # Deep Amber
    # ('oranye',  [155, 65,  15]),   # Burnt Amber
    # ('oranye',  [150, 62,  14]),   # Amber Brown Light
    # ('oranye',  [146, 64,  14]),   # Amber Brown
    # ('oranye',  [140, 58,  14]),   # Amber Brown Dark
    # ('oranye',  [135, 56,  14]),   # Dark Copper
    # ('oranye',  [130, 55,  15]),   # Russet
    # ('oranye',  [125, 53,  15]),   # Copper Brown
    # ('oranye',  [120, 53,  15]),   # Amber Brown Tua

    # ── KUNING ─────────────────────────────────────────────────────────────────
    ('kuning',  [255, 255, 160]),  # Light Yellow Pastel
    ('kuning',  [255, 250, 100]),  # Pale Yellow
    ('kuning',  [255, 245, 80]),   # Pastel Yellow
    ('kuning',  [255, 240, 60]),   # Canary
    ('kuning',  [255, 235, 50]),   # Lemon Yellow
    ('kuning',  [255, 228, 40]),   # Bright Yellow
    ('kuning',  [253, 224, 71]),   # Light Yellow
    ('kuning',  [252, 215, 60]),   # Banana Yellow
    ('kuning',  [252, 211, 77]),   # Light Amber
    ('kuning',  [250, 200, 55]),   # Maize
    ('kuning',  [248, 195, 50]),   # Cream Yellow Deep
    ('kuning',  [245, 185, 40]),   # School Bus Yellow
    ('kuning',  [243, 180, 30]),   # Saffron Light
    ('kuning',  [241, 175, 25]),   # Aureolin
    ('kuning',  [240, 170, 20]),   # Naples Yellow
    ('kuning',  [238, 168, 18]),   # Amber Light
    ('kuning',  [236, 162, 15]),   # Yellow Ochre Light
    ('kuning',  [234, 179, 8]),    # Yellow
    ('kuning',  [232, 158, 12]),   # Gold Light
    ('kuning',  [230, 190, 60]),   # Medium Gold
    ('kuning',  [228, 155, 10]),   # Old Gold
    ('kuning',  [225, 170, 40]),   # Buff Yellow
    ('kuning',  [222, 160, 30]),   # Saffron
    ('kuning',  [220, 155, 25]),   # Dark Amber Light
    ('kuning',  [218, 165, 32]),   # Goldenrod
    ('kuning',  [215, 155, 25]),   # Deep Gold
    ('kuning',  [212, 150, 22]),   # Gold
    ('kuning',  [210, 145, 20]),   # Harvest Gold
    ('kuning',  [210, 160, 50]),   # Dark Gold
    ('kuning',  [208, 140, 18]),   # Old Gold Dark
    ('kuning',  [205, 155, 45]),   # Dark Straw
    ('kuning',  [200, 180, 80]),   # Muted Gold
    ('kuning',  [198, 140, 18]),   # Muted Amber
    ('kuning',  [195, 138, 16]),   # Dark Yellow Ochre
    ('kuning',  [190, 135, 20]),   # Khaki Yellow
    ('kuning',  [188, 132, 18]),   # Yellow Brown Light
    ('kuning',  [185, 128, 16]),   # Dull Gold
    ('kuning',  [182, 125, 14]),   # Antique Gold
    ('kuning',  [180, 155, 60]),   # Muted Dark Mustard
    ('kuning',  [178, 120, 12]),   # Dark Mustard
    ('kuning',  [175, 118, 12]),   # Mustard
    ('kuning',  [170, 115, 10]),   # Ochre
    ('kuning',  [165, 110, 10]),   # Dark Ochre
    ('kuning',  [160, 105, 10]),   # Olive Gold
    ('kuning',  [155, 100, 8]),    # Dark Olive Gold
    ('kuning',  [150, 95,  8]),    # Very Dark Mustard

    # ── COKLAT ─────────────────────────────────────────────────────────────────
    ('coklat',  [210, 180, 140]),  # Tan
    ('coklat',  [205, 170, 128]),  # Light Tan
    ('coklat',  [200, 162, 118]),  # Wheat Brown
    ('coklat',  [195, 155, 108]),  # Sandy Brown Light
    ('coklat',  [190, 148, 100]),  # Sandy Brown
    ('coklat',  [185, 142, 90]),   # Pale Brown
    ('coklat',  [180, 135, 82]),   # Buff Brown
    ('coklat',  [175, 128, 72]),   # Dark Tan
    ('coklat',  [170, 130, 80]),   # Sandy Brown Dark
    ('coklat',  [165, 120, 70]),   # Peanut Butter
    ('coklat',  [160, 115, 65]),   # Brown Light
    ('coklat',  [160, 100, 50]),   # Ochre Brown
    ('coklat',  [155, 110, 60]),   # Brown Medium Light
    ('coklat',  [152, 118, 84]),   # Light Wood Brown
    ('coklat',  [150, 120, 95]),   # Light Brown
    ('coklat',  [147, 112, 75]),   # Wood Brown
    ('coklat',  [143, 106, 70]),   # Medium Brown Light
    ('coklat',  [140, 110, 70]),   # Caramel Brown
    ('coklat',  [138, 100, 62]),   # Toffee
    ('coklat',  [135, 110, 90]),   # Grayish Brown
    ('coklat',  [133, 95,  55]),   # Caramel
    ('coklat',  [130, 90,  50]),   # Hazel Brown
    ('coklat',  [128, 92,  64]),   # Raw Umber Light
    ('coklat',  [125, 100, 80]),   # Warm Taupe
    ('coklat',  [122, 88,  58]),   # Umber Brown
    ('coklat',  [120, 85,  52]),   # Medium Umber
    ('coklat',  [118, 82,  48]),   # Walnut Light
    ('coklat',  [115, 80,  45]),   # Walnut
    ('coklat',  [112, 78,  42]),   # Dark Walnut
    ('coklat',  [110, 90,  75]),   # Dark Taupe
    ('coklat',  [110, 80,  60]),   # Medium Brown Terang
    ('coklat',  [108, 75,  40]),   # Brown Bistre
    ('coklat',  [105, 72,  38]),   # Dark Brown Light
    ('coklat',  [103, 68,  35]),   # Deep Umber
    ('coklat',  [100, 70,  50]),   # Medium Brown
    ('coklat',  [98,  65,  32]),   # Chocolate Brown Light
    ('coklat',  [96,  62,  28]),   # Chocolate
    ('coklat',  [95,  75,  55]),   # Coklat Sogan Khas Batik
    ('coklat',  [92,  60,  28]),   # Dark Chocolate Light
    ('coklat',  [90,  60,  40]),   # Medium Brown Gelap
    ('coklat',  [88,  58,  26]),   # Dark Chocolate
    ('coklat',  [85,  55,  28]),   # Very Dark Chocolate
    ('coklat',  [82,  52,  25]),   # Brown Black Light
    ('coklat',  [80,  60,  45]),   # Coklat Sogan Gelap
    ('coklat',  [80,  52,  30]),   # Medium Dark Brown
    ('coklat',  [75,  48,  25]),   # Deep Brown
    ('coklat',  [70,  45,  25]),   # Dark Brown
    ('coklat',  [65,  40,  20]),   # Very Dark Brown
    ('coklat',  [60,  35,  18]),   # Dark Brown Pekat
    ('coklat',  [52,  28,  14]),   # Brown Black
    ('coklat',  [45,  25,  15]),   # Ultra Dark Brown

    # ── BIRU ───────────────────────────────────────────────────────────────────
    ('biru',    [200, 220, 255]),  # Alice Blue
    ('biru',    [180, 210, 255]),  # Lavender Blue
    ('biru',    [160, 200, 255]),  # Baby Blue Light
    ('biru',    [145, 190, 255]),  # Pale Blue
    ('biru',    [130, 180, 255]),  # Light Sky Blue
    ('biru',    [115, 170, 255]),  # Cornflower Blue Light
    ('biru',    [100, 160, 248]),  # Soft Blue
    ('biru',    [85,  148, 248]),  # Periwinkle
    ('biru',    [56,  189, 248]),  # Sky Blue
    ('biru',    [59,  130, 246]),  # Blue
    ('biru',    [50,  120, 240]),  # Dodger Blue
    ('biru',    [45,  112, 230]),  # Cornflower Blue
    ('biru',    [40,  100, 220]),  # Vivid Blue
    ('biru',    [35,  90,  210]),  # Medium Blue
    ('biru',    [29,  78,  216]),  # Royal Blue
    ('biru',    [25,  72,  200]),  # Strong Blue
    ('biru',    [20,  65,  185]),  # Blue Medium Dark
    ('biru',    [2,   132, 199]),  # Cerulean
    ('biru',    [15,  55,  165]),  # Dark Cerulean
    ('biru',    [12,  48,  150]),  # Dark Blue
    ('biru',    [10,  42,  138]),  # Marine Blue
    ('biru',    [30,  58,  138]),  # Navy Blue
    ('biru',    [8,   38,  125]),  # Strong Navy
    ('biru',    [40,  70,  110]),  # Medium Navy
    ('biru',    [6,   32,  112]),  # Dark Navy
    ('biru',    [5,   28,  98]),   # Deep Navy
    ('biru',    [5,   24,  85]),   # Very Deep Navy
    ('biru',    [4,   20,  72]),   # Ultra Deep Navy
    ('biru',    [20,  30,  60]),   # Dark Navy
    ('biru',    [15,  40,  80]),   # Deep Navy 2
    ('biru',    [10,  20,  40]),   # Ultra Dark Navy
    ('biru',    [49,  46,  129]),  # Indigo Dark
    ('biru',    [70,  90,  120]),  # Slate Blue
    ('biru',    [85,  105, 135]),  # Slate Blue Light
    ('biru',    [100, 120, 150]),  # Muted Blue
    ('biru',    [120, 138, 165]),  # Steel Blue Muted

    # ── HIJAU ──────────────────────────────────────────────────────────────────
    ('hijau',   [200, 245, 215]),  # Mint Green Pale
    ('hijau',   [180, 235, 200]),  # Honeydew
    ('hijau',   [160, 225, 185]),  # Light Mint
    ('hijau',   [140, 215, 170]),  # Pale Green
    ('hijau',   [120, 205, 155]),  # Light Green Soft
    ('hijau',   [100, 200, 140]),  # Soft Green
    ('hijau',   [74,  222, 128]),  # Light Green
    ('hijau',   [60,  210, 115]),  # Emerald Light
    ('hijau',   [34,  197, 94]),   # Green
    ('hijau',   [25,  180, 80]),   # Medium Green
    ('hijau',   [20,  170, 70]),   # Vivid Green
    ('hijau',   [18,  160, 60]),   # Strong Green
    ('hijau',   [21,  128, 61]),   # Forest Green
    ('hijau',   [18,  115, 55]),   # Medium Forest Green
    ('hijau',   [15,  102, 48]),   # Dark Forest Green 2
    ('hijau',   [20,  83,  45]),   # Dark Forest Green
    ('hijau',   [16,  75,  40]),   # Deep Forest Green
    ('hijau',   [14,  65,  35]),   # Pine Green
    ('hijau',   [12,  55,  30]),   # Hunter Green
    ('hijau',   [10,  48,  25]),   # Very Dark Green
    ('hijau',   [20,  40,  20]),   # Ultra Dark Green
    ('hijau',   [15,  30,  15]),   # Ultra Dark Green 2
    ('hijau',   [101, 163, 13]),   # Lime Green
    ('hijau',   [85,  145, 12]),   # Lime Green Dark
    ('hijau',   [70,  128, 10]),   # Medium Lime Green
    ('hijau',   [80,  130, 60]),   # Medium Olive Green
    ('hijau',   [70,  120, 55]),   # Dark Olive Green Light
    ('hijau',   [60,  110, 48]),   # Avocado Green
    ('hijau',   [50,  100, 42]),   # Fern Green
    ('hijau',   [60,  80,  50]),   # Dark Olive Green
    ('hijau',   [70,  100, 70]),   # Muted Forest Green
    ('hijau',   [80,  110, 80]),   # Sage Green
    ('hijau',   [90,  110, 80]),   # Grayish Olive
    ('hijau',   [40,  90,  90]),   # Dark Teal
    ('hijau',   [35,  80,  80]),   # Dark Teal Deep
    ('hijau',   [30,  70,  70]),   # Teal Very Dark
    ('hijau',   [20,  184, 166]),  # Teal
    ('hijau',   [15,  118, 110]),  # Dark Teal Tosca
    ('hijau',   [12,  95,  88]),   # Deep Teal
    ('hijau',   [50,  110, 110]),  # Dark Grayish Cyan
    ('hijau',   [90,  150, 150]),  # Grayish Cyan
    ('hijau',   [120, 170, 170]),  # Light Muted Cyan

    # ── UNGU ───────────────────────────────────────────────────────────────────
    ('ungu',    [230, 190, 255]),  # Pale Lavender
    ('ungu',    [215, 170, 252]),  # Lavender Light
    ('ungu',    [200, 150, 250]),  # Mauve Light
    ('ungu',    [185, 130, 248]),  # Medium Lavender
    ('ungu',    [170, 110, 248]),  # Lavender Medium
    ('ungu',    [168, 85,  247]),  # Light Purple
    ('ungu',    [155, 75,  245]),  # Violet Light
    ('ungu',    [145, 65,  235]),  # Medium Violet
    ('ungu',    [135, 55,  225]),  # Soft Violet
    ('ungu',    [130, 80,  130]),  # Dusty Lavender
    ('ungu',    [126, 34,  206]),  # Purple
    ('ungu',    [118, 28,  195]),  # Rich Purple
    ('ungu',    [110, 25,  180]),  # Medium Purple
    ('ungu',    [110, 30,  90]),   # Deep Violet Red
    ('ungu',    [105, 22,  168]),  # Dark Purple Light
    ('ungu',    [100, 20,  155]),  # Violet
    ('ungu',    [100, 35,  80]),   # Magenta Tua
    ('ungu',    [95,  18,  145]),  # Dark Violet Light
    ('ungu',    [90,  30,  70]),   # Deep Violet Red
    ('ungu',    [90,  15,  135]),  # Dark Violet
    ('ungu',    [88,  28,  135]),  # Dark Purple
    ('ungu',    [85,  15,  125]),  # Grape
    ('ungu',    [82,  12,  115]),  # Eggplant Light
    ('ungu',    [80,  25,  65]),   # Dark Violet 2
    ('ungu',    [78,  10,  105]),  # Eggplant
    ('ungu',    [75,  25,  60]),   # Aubergine
    ('ungu',    [72,  8,   98]),   # Deep Eggplant
    ('ungu',    [68,  8,   90]),   # Byzantium Light
    ('ungu',    [65,  5,   82]),   # Byzantium
    ('ungu',    [62,  20,  58]),   # Dark Plum 2
    ('ungu',    [60,  20,  55]),   # Deep Plum
    ('ungu',    [58,  5,   72]),   # Royal Purple Dark
    ('ungu',    [55,  5,   65]),   # Dark Byzantium
    ('ungu',    [50,  15,  40]),   # Dark Plum
    ('ungu',    [45,  12,  50]),   # Tyrian Purple
    ('ungu',    [40,  10,  45]),   # Dark Tyrian
    ('ungu',    [40,  15,  40]),   # Ultra Dark Purple
    ]

    def __init__(self):
        self.TARGET_LAB_LIST = self.get_target_lab_list()

    @staticmethod
    def get_closest_color_name(rgb):
        r, g, b = [x / 255.0 for x in rgb]
        min_dist = float('inf')
        closest_name = '-'
        for name, hex_val in mcolors.CSS4_COLORS.items():
            rc, gc, bc = mcolors.to_rgb(hex_val)
            dist = (r - rc)**2 + (g - gc)**2 + (b - bc)**2
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        return closest_name
    
    def get_target_lab_list(self):
        result = []
        for name, rgb in self.TARGET_COLORS_REF:
            rgb_arr = np.array([rgb], dtype=np.float32) / 255.0
            lab_val = skcolor.rgb2lab(rgb_arr)[0]   # shape (3,): [L*, a*, b*]
            result.append((name, lab_val))
        return result
    
    def get_label_color(self, L_n: float, a_n: float, b_n: float) -> str:
        L_star = L_n * 100.0
        a_star = a_n * 255.0 - 128.0
        b_star = b_n * 255.0 - 128.0
        input_lab = np.array([L_star, a_star, b_star])

        min_dist = float('inf')
        closest  = 'abu-abu'
        for name, target_lab in self.TARGET_LAB_LIST:
            d = deltaE_ciede2000(input_lab, target_lab)
            if d < min_dist:
                min_dist = d
                closest  = name
        return closest
    
    def lab_to_hex(self, L_n: float, a_n: float, b_n: float) -> str:
        L_star = L_n * 100.0
        a_star = a_n * 255.0 - 128.0
        b_star = b_n * 255.0 - 128.0
        input_lab = np.array([[[L_star, a_star, b_star]]])
        rgb = skcolor.lab2rgb(input_lab)[0][0]
        # lab2rgb returns values usually in [0, 1]
        r = min(max(int(round(rgb[0] * 255)), 0), 255)
        g = min(max(int(round(rgb[1] * 255)), 0), 255)
        b = min(max(int(round(rgb[2] * 255)), 0), 255)
        return f"#{r:02x}{g:02x}{b:02x}"
