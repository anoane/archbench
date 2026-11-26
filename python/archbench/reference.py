# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent, slow Python numeric references. Never reported as GPU results."""
from __future__ import annotations
import bisect
import functools
import math
import struct

NF4 = (-1.0, -.6961928009986877, -.5250730514526367, -.39491748809814453,
       -.28444138169288635, -.18477343022823334, -.09105003625154495, 0.0,
       .07958029955625534, .16093020141124725, .24611230194568634, .33791524171829224,
       .44070982933044434, .5626170039176941, .7229568362236023, 1.0)
# exponent bits, fraction bits, bias, special-value convention
FLOATS = {"e2m1": (2, 1, 1, "finite"), "e2m3": (2, 3, 1, "finite"),
          "e3m2": (3, 2, 3, "finite"), "e4m3": (4, 3, 7, "terminal_nan"),
          "e5m2": (5, 2, 15, "ieee"), "f16": (5, 10, 15, "ieee"),
          "bf16": (8, 7, 127, "ieee")}


def f32(value):
    try: return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError: return math.copysign(math.inf, value)


def width(name):
    if name == "nf4": return 4
    if name in FLOATS:
        e, m, _, _ = FLOATS[name]
        return 1 + e + m
    if len(name) > 1 and name[0] in "su" and name[1:].isdigit() and 1 <= int(name[1:]) <= 64:
        return int(name[1:])
    raise ValueError(f"unknown format {name}")


def decode(code, name):
    bits = width(name)
    if not 0 <= code < 2 ** bits: raise ValueError("code out of range")
    if name == "nf4": return NF4[code]
    if name not in FLOATS:
        return code - 2 ** bits if name[0] == "s" and code >= 2 ** (bits - 1) else code
    e, m, bias, special = FLOATS[name]
    sign = -1.0 if code & (1 << (e + m)) else 1.0
    exponent = (code >> m) & (2 ** e - 1)
    fraction = code & (2 ** m - 1)
    if exponent == 2 ** e - 1:
        if special == "ieee": return sign * (math.nan if fraction else math.inf)
        if special == "terminal_nan" and fraction == 2 ** m - 1: return math.nan
    if exponent == 0: return sign * math.ldexp(fraction, 1 - bias - m)
    return sign * math.ldexp(1 + fraction / 2 ** m, exponent - bias)


@functools.lru_cache(None)
def positive_table(name):
    if name == "nf4": return NF4
    return tuple(value for code in range(2 ** (width(name) - 1))
                 if math.isfinite(value := decode(code, name)))


def encode(value, name):
    bits = width(name)
    if math.isnan(value):
        if name in FLOATS and FLOATS[name][3] != "finite": return 2 ** (bits - 1) - 1
        return 7 if name == "nf4" else 0
    if name not in FLOATS and name != "nf4":
        low = -2 ** (bits - 1) if name[0] == "s" else 0
        high = 2 ** (bits - (name[0] == "s")) - 1
        return round(min(high, max(low, value))) % 2 ** bits
    if name == "nf4":
        clipped = min(1, max(-1, value))
        return min(range(16), key=lambda i: (abs(clipped - NF4[i]), i % 2, i))
    negative = math.copysign(1, value) < 0
    value = abs(value)
    table = positive_table(name)
    position = bisect.bisect_left(table, value)
    candidates = [i for i in (position - 1, position) if 0 <= i < len(table)]
    if math.isinf(value): chosen = len(table) - 1
    else: chosen = min(candidates, key=lambda i: (abs(value - table[i]), i % 2, i))
    return chosen | (negative << (bits - 1))


def pack(codes, bits):
    if not 1 <= bits <= 64: raise ValueError("bits must be 1..64")
    codes = list(codes)
    value = 0
    for index, code in enumerate(codes):
        if not 0 <= code < 2 ** bits: raise ValueError("code outside width")
        value |= code << (index * bits)
    return value.to_bytes((len(codes) * bits + 7) // 8, "little")


def unpack(data, count, bits):
    if not 1 <= bits <= 64 or count < 0: raise ValueError("bad count/width")
    if len(data) < (count * bits + 7) // 8: raise ValueError("truncated packed stream")
    value = int.from_bytes(data, "little")
    return [(value >> (index * bits)) & (2 ** bits - 1) for index in range(count)]


def modular_integer(a, b, bits, signed, op):
    if not 1 <= bits <= 64: raise ValueError("width 1..64")
    modulus = 2 ** bits
    a %= modulus; b %= modulus
    signed_a = a - modulus if signed and a >= modulus // 2 else a
    signed_b = b - modulus if signed and b >= modulus // 2 else b
    shift = b % bits
    operations = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b,
                  "and": lambda: a & b, "or": lambda: a | b, "xor": lambda: a ^ b,
                  "shl": lambda: a << shift, "shr": lambda: signed_a >> shift,
                  "min": lambda: min(signed_a, signed_b), "max": lambda: max(signed_a, signed_b)}
    return operations[op]() % modulus


def scale_e8m0(value):
    if not value > 0: return 0
    mantissa, exponent = math.frexp(value)
    return min(254, max(0, exponent - (mantissa == .5) + 127))


def quantize(values, name, group=32, scale_format="f32"):
    if group < 1: raise ValueError("group must be positive")
    bits = width(name)
    payload, scales, offsets, decoded = bytearray(), [], [], []
    for start in range(0, len(values), group):
        block = [f32(x) for x in values[start:start + group]]
        if any(not math.isfinite(x) for x in block): raise ValueError("block quantizer reference expects finite inputs")
        affine = name.startswith("u")
        offset = min(block) if affine else 0.0
        maximum = max(positive_table(name)) if name in FLOATS or name == "nf4" else (2 ** (bits - (name[0] == "s")) - 1)
        scale = f32(f32(max(block) - min(block)) / max(1, maximum)) if affine else f32(max(map(abs, block)) / max(1, maximum))
        if not scale > 0 or name in ("f16", "bf16"): scale = 1.0
        if scale_format == "e8m0": scale = math.ldexp(1.0, scale_e8m0(scale) - 127)
        elif scale_format == "e4m3": scale = max(2 ** -9, decode(encode(scale, "e4m3"), "e4m3"))
        elif scale_format != "f32": raise ValueError("unknown scale format")
        codes = [encode(f32(f32(x - offset) / scale), name) for x in block]
        packed = pack(codes, bits)
        payload.extend(packed + bytes((group * bits + 7) // 8 - len(packed)))
        scales.append(scale); offsets.append(offset)
        decoded.extend(f32(f32(decode(code, name) * scale) + offset) for code in codes)
    return {"packed": bytes(payload), "scales": scales, "offsets": offsets, "decoded": decoded}
