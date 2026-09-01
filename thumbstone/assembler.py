"""A two-pass assembler for the subset the core executes.

Hand-encoding test programs as hex was miserable and produced bugs that
looked like emulator bugs, which is the worst kind of time sink. This exists
so the test programs read as assembly and the diff when something breaks is
in one place.

Pass one walks the source assigning addresses and collecting labels; pass
two encodes. Branch ranges are checked and reported with the source line,
because a silently truncated offset is a debugging afternoon.
"""

import re

from .cpu import MASK32

LOW_REGS = {"r%d" % i: i for i in range(8)}
ALL_REGS = {"r%d" % i: i for i in range(16)}
ALL_REGS.update({"sp": 13, "lr": 14, "pc": 15})

ALU_OPS = {
    "ands": 0x0, "eors": 0x1, "lsls": 0x2, "lsrs": 0x3, "asrs": 0x4,
    "adcs": 0x5, "sbcs": 0x6, "rors": 0x7, "tst": 0x8, "rsbs": 0x9,
    "cmp": 0xA, "cmn": 0xB, "orrs": 0xC, "muls": 0xD, "bics": 0xE, "mvns": 0xF,
}

CONDS = {"eq": 0, "ne": 1, "cs": 2, "hs": 2, "cc": 3, "lo": 3, "mi": 4, "pl": 5,
         "vs": 6, "vc": 7, "hi": 8, "ls": 9, "ge": 10, "lt": 11, "gt": 12, "le": 13}


class AsmError(Exception):
    def __init__(self, line_no, text, message):
        super().__init__("line %d: %s  (%s)" % (line_no, message, text.strip()))
        self.line_no = line_no


def _reg(tok, line_no, text, low_only=True):
    t = tok.strip().lower()
    table = LOW_REGS if low_only else ALL_REGS
    if t not in table:
        raise AsmError(line_no, text, "expected %s register, got '%s'" % ("low" if low_only else "a", tok))
    return table[t]


def _imm(tok, line_no, text):
    t = tok.strip().lstrip("#")
    try:
        return int(t, 0)
    except ValueError:
        raise AsmError(line_no, text, "bad immediate '%s'" % tok)


def _split_args(rest):
    """Split on commas that are not inside a register list or an address.

    Tracking only braces was not enough: '[r7, #0]' split down the middle and
    every immediate-offset load silently took the wrong assembler path.
    """
    if not rest:
        return []
    parts = []
    depth = 0
    cur = ""
    for ch in rest:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip() != ""]


def assemble(source, origin=0):
    """Returns (bytes, labels). Addresses are absolute from `origin`."""
    lines = source.split("\n")
    cleaned = []
    for idx, raw in enumerate(lines):
        text = raw.split(";")[0].split("//")[0].rstrip()
        if not text.strip():
            continue
        cleaned.append((idx + 1, text))

    # pass one: sizes and labels
    labels = {}
    addr = origin
    layout = []
    for line_no, text in cleaned:
        body = text.strip()
        # A line may carry several labels before an instruction, or be a
        # label on its own.
        while ":" in body and body.index(":") < (body.index(" ") if " " in body else len(body)):
            label, _, remainder = body.partition(":")
            labels[label.strip()] = addr
            body = remainder.strip()
        if not body:
            continue
        mnemonic = body.split()[0].lower()
        if mnemonic == ".word":
            count = len(_split_args(body[len(mnemonic):]))
            size = 4 * count
        elif mnemonic == ".align":
            pad = (-addr) % 4
            size = pad
        elif mnemonic == "bl":
            size = 4
        else:
            size = 2
        layout.append((line_no, text, body, addr, size))
        addr += size

    # pass two: encode
    out = bytearray()

    def emit16(value):
        out.append(value & 0xFF)
        out.append((value >> 8) & 0xFF)

    def emit32(value):
        emit16(value & 0xFFFF)
        emit16((value >> 16) & 0xFFFF)

    for line_no, text, body, at, size in layout:
        parts = body.split(None, 1)
        mnemonic = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        args = _split_args(rest)

        def resolve(tok):
            t = tok.strip()
            if t in labels:
                return labels[t]
            return _imm(t, line_no, text)

        if mnemonic == ".word":
            for a in args:
                emit32(resolve(a) & MASK32)
            continue
        if mnemonic == ".align":
            for _ in range(size):
                out.append(0)
            continue

        if mnemonic in ("movs", "cmp", "adds", "subs") and len(args) == 2 and args[1].strip().startswith("#"):
            rd = _reg(args[0], line_no, text)
            imm = _imm(args[1], line_no, text)
            if not 0 <= imm <= 255:
                raise AsmError(line_no, text, "8-bit immediate out of range")
            kind = {"movs": 0, "cmp": 1, "adds": 2, "subs": 3}[mnemonic]
            emit16(0x2000 | (kind << 11) | (rd << 8) | imm)
            continue

        if mnemonic in ("lsls", "lsrs", "asrs") and len(args) == 3 and args[2].strip().startswith("#"):
            rd = _reg(args[0], line_no, text)
            rm = _reg(args[1], line_no, text)
            imm = _imm(args[2], line_no, text)
            if not 0 <= imm <= 31:
                raise AsmError(line_no, text, "shift amount out of range")
            kind = {"lsls": 0, "lsrs": 1, "asrs": 2}[mnemonic]
            emit16((kind << 11) | (imm << 6) | (rm << 3) | rd)
            continue

        if mnemonic in ("adds", "subs") and len(args) == 3:
            rd = _reg(args[0], line_no, text)
            rn = _reg(args[1], line_no, text)
            third = args[2].strip()
            is_sub = 1 if mnemonic == "subs" else 0
            if third.startswith("#"):
                imm = _imm(third, line_no, text)
                if not 0 <= imm <= 7:
                    raise AsmError(line_no, text, "3-bit immediate out of range")
                emit16(0x1C00 | (is_sub << 9) | (imm << 6) | (rn << 3) | rd)
            else:
                rm = _reg(third, line_no, text)
                emit16(0x1800 | (is_sub << 9) | (rm << 6) | (rn << 3) | rd)
            continue

        if mnemonic in ("add", "sub") and len(args) == 2 and args[0].strip().lower() == "sp":
            imm = _imm(args[1], line_no, text)
            if imm % 4 or not 0 <= imm <= 508:
                raise AsmError(line_no, text, "sp adjustment must be a multiple of 4 up to 508")
            emit16(0xB000 | ((1 if mnemonic == "sub" else 0) << 7) | (imm >> 2))
            continue

        if mnemonic == "add" and len(args) == 3 and args[1].strip().lower() in ("sp", "pc"):
            rd = _reg(args[0], line_no, text)
            imm = _imm(args[2], line_no, text)
            if imm % 4 or not 0 <= imm <= 1020:
                raise AsmError(line_no, text, "offset must be a multiple of 4 up to 1020")
            is_sp = 1 if args[1].strip().lower() == "sp" else 0
            emit16(0xA000 | (is_sp << 11) | (rd << 8) | (imm >> 2))
            continue

        if mnemonic in ("add", "mov") and len(args) == 2:
            rd = _reg(args[0], line_no, text, low_only=False)
            rm = _reg(args[1], line_no, text, low_only=False)
            code = 0 if mnemonic == "add" else 2
            emit16(0x4400 | (code << 8) | ((rd >> 3) << 7) | (rm << 3) | (rd & 7))
            continue

        if mnemonic == "cmp" and len(args) == 2 and (
                args[0].strip().lower() not in LOW_REGS or args[1].strip().lower() not in LOW_REGS):
            rd = _reg(args[0], line_no, text, low_only=False)
            rm = _reg(args[1], line_no, text, low_only=False)
            emit16(0x4500 | ((rd >> 3) << 7) | (rm << 3) | (rd & 7))
            continue

        if mnemonic in ALU_OPS and len(args) == 2:
            rd = _reg(args[0], line_no, text)
            rm = _reg(args[1], line_no, text)
            emit16(0x4000 | (ALU_OPS[mnemonic] << 6) | (rm << 3) | rd)
            continue

        if mnemonic == "bx" and len(args) == 1:
            rm = _reg(args[0], line_no, text, low_only=False)
            emit16(0x4700 | (rm << 3))
            continue

        if mnemonic in ("ldr", "str", "ldrb", "strb", "ldrh", "strh") and len(args) == 2:
            rd = _reg(args[0], line_no, text)
            mem = args[1].strip()
            m = re.match(r"^\[\s*(\w+)\s*(?:,\s*(#?-?\w+)\s*)?\]$", mem)
            if not m:
                # ldr rd, =label style literal pool is not supported; use .word
                raise AsmError(line_no, text, "expected [rn] or [rn, #imm]")
            base_tok, off_tok = m.group(1), m.group(2)
            offset = 0 if off_tok is None else (
                labels[off_tok] if off_tok in labels else _imm(off_tok, line_no, text))
            if base_tok.lower() == "sp":
                if mnemonic not in ("ldr", "str"):
                    raise AsmError(line_no, text, "only ldr/str support sp-relative form")
                if offset % 4 or not 0 <= offset <= 1020:
                    raise AsmError(line_no, text, "sp offset must be a multiple of 4 up to 1020")
                emit16(0x9000 | ((1 if mnemonic == "ldr" else 0) << 11) | (rd << 8) | (offset >> 2))
                continue
            if base_tok.lower() == "pc":
                if mnemonic != "ldr":
                    raise AsmError(line_no, text, "only ldr supports pc-relative form")
                if offset % 4 or not 0 <= offset <= 1020:
                    raise AsmError(line_no, text, "pc offset must be a multiple of 4 up to 1020")
                emit16(0x4800 | (rd << 8) | (offset >> 2))
                continue
            rn = _reg(base_tok, line_no, text)
            if mnemonic in ("ldr", "str"):
                if offset % 4 or not 0 <= offset <= 124:
                    raise AsmError(line_no, text, "word offset must be a multiple of 4 up to 124")
                emit16(0x6000 | ((1 if mnemonic == "ldr" else 0) << 11) | ((offset >> 2) << 6) | (rn << 3) | rd)
            elif mnemonic in ("ldrb", "strb"):
                if not 0 <= offset <= 31:
                    raise AsmError(line_no, text, "byte offset out of range")
                emit16(0x7000 | ((1 if mnemonic == "ldrb" else 0) << 11) | (offset << 6) | (rn << 3) | rd)
            else:
                if offset % 2 or not 0 <= offset <= 62:
                    raise AsmError(line_no, text, "halfword offset must be even and at most 62")
                emit16(0x8000 | ((1 if mnemonic == "ldrh" else 0) << 11) | ((offset >> 1) << 6) | (rn << 3) | rd)
            continue

        if mnemonic in ("push", "pop") and len(args) == 1:
            inner = args[0].strip()
            if not (inner.startswith("{") and inner.endswith("}")):
                raise AsmError(line_no, text, "expected a register list in braces")
            names = [n.strip().lower() for n in inner[1:-1].split(",") if n.strip()]
            mask = 0
            extra = 0
            for n in names:
                if n in ("lr", "pc"):
                    extra = 1
                    continue
                if n not in LOW_REGS:
                    raise AsmError(line_no, text, "push/pop only handle r0-r7 plus lr/pc")
                mask |= 1 << LOW_REGS[n]
            is_pop = 1 if mnemonic == "pop" else 0
            emit16(0xB400 | (is_pop << 11) | (extra << 8) | mask)
            continue

        if mnemonic in ("cpsid", "cpsie"):
            # CPSIE i is 0xB662 and CPSID i is 0xB672; the difference is the
            # 'im' bit at position 4, and the low nibble 2 selects PRIMASK.
            emit16(0xB672 if mnemonic == "cpsid" else 0xB662)
            continue

        if mnemonic == "svc":
            emit16(0xDF00 | (resolve(args[0]) & 0xFF))
            continue

        if mnemonic == "nop":
            emit16(0x46C0)  # mov r8, r8
            continue

        if mnemonic == "bl":
            target = resolve(args[0])
            offset = target - (at + 4)
            if offset % 2:
                raise AsmError(line_no, text, "branch target is not halfword aligned")
            if not -(1 << 24) <= offset < (1 << 24):
                raise AsmError(line_no, text, "bl target out of range")
            s = 1 if offset < 0 else 0
            val = offset & 0x1FFFFFF
            imm11 = (val >> 1) & 0x7FF
            imm10 = (val >> 12) & 0x3FF
            i2 = (val >> 22) & 1
            i1 = (val >> 23) & 1
            # The decoder computes i = 1 - (j xor s); inverting that gives
            # j = (1 - i) xor s.
            j1 = ((1 - i1) ^ s) & 1
            j2 = ((1 - i2) ^ s) & 1
            emit16(0xF000 | (s << 10) | imm10)
            emit16(0xD000 | (j1 << 13) | (j2 << 11) | imm11)
            continue

        if mnemonic == "b" or (mnemonic.startswith("b") and mnemonic[1:] in CONDS):
            target = at + 4 if args and args[0].strip() == "." else resolve(args[0])
            if args and args[0].strip() == ".":
                target = at
            offset = target - (at + 4)
            if offset % 2:
                raise AsmError(line_no, text, "branch target is not halfword aligned")
            if mnemonic == "b":
                imm = offset >> 1
                if not -1024 <= imm <= 1023:
                    raise AsmError(line_no, text, "unconditional branch out of range")
                emit16(0xE000 | (imm & 0x7FF))
            else:
                imm = offset >> 1
                if not -128 <= imm <= 127:
                    raise AsmError(line_no, text, "conditional branch out of range (+/-256 bytes)")
                emit16(0xD000 | (CONDS[mnemonic[1:]] << 8) | (imm & 0xFF))
            continue

        raise AsmError(line_no, text, "cannot assemble '%s'" % mnemonic)

    return bytes(out), labels

