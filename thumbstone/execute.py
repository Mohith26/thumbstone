"""Decode and execute the 16-bit Thumb subset, plus 32-bit BL.

The decoder is a straight if/elif ladder over the top bits, ordered the way
the ARMv6-M encoding tables are laid out. It is not the fastest possible
shape (a jump table on the top byte would beat it) but it stays readable
next to the manual, which matters more when you are chasing a flag bug.

Flag semantics are the part worth being pedantic about:
  * ADD sets C on unsigned carry out, V on signed overflow.
  * SUB is implemented as a + ~b + 1 so C means "no borrow", matching the
    architecture rather than the intuitive reading.
  * Shifts write C from the last bit shifted out; a shift of zero leaves C
    untouched, and LSR/ASR #0 mean shift by 32, which is a real encoding
    quirk rather than a typo.
"""

from .cpu import MASK32, SP, LR, PC, MemoryFault, SYSTICK_EXCEPTION

COND_NAMES = ["eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc",
              "hi", "ls", "ge", "lt", "gt", "le", "al", "nv"]


def sign32(v):
    v &= MASK32
    return v - (1 << 32) if v & 0x80000000 else v


def add_with_carry(a, b, carry_in):
    """Returns (result, carry_out, overflow) exactly as the ARM pseudocode."""
    a &= MASK32
    b &= MASK32
    usum = a + b + (1 if carry_in else 0)
    result = usum & MASK32
    carry_out = usum > MASK32
    ssum = sign32(a) + sign32(b) + (1 if carry_in else 0)
    overflow = sign32(result) != ssum
    return result, carry_out, overflow


def lsl_c(value, amount):
    value &= MASK32
    if amount == 0:
        return value, None
    if amount >= 33:
        return 0, False
    shifted = value << amount
    return shifted & MASK32, bool((shifted >> 32) & 1)


def lsr_c(value, amount):
    value &= MASK32
    if amount == 0:
        return value, None
    if amount >= 33:
        return 0, False
    return (value >> amount) & MASK32, bool((value >> (amount - 1)) & 1)


def asr_c(value, amount):
    value &= MASK32
    if amount == 0:
        return value, None
    signed = sign32(value)
    if amount >= 32:
        result = 0 if signed >= 0 else MASK32
        return result, bool(signed < 0)
    return (signed >> amount) & MASK32, bool((signed >> (amount - 1)) & 1)


def ror_c(value, amount):
    value &= MASK32
    if amount == 0:
        return value, None
    amount &= 31
    if amount == 0:
        return value, bool((value >> 31) & 1)
    result = ((value >> amount) | (value << (32 - amount))) & MASK32
    return result, bool((result >> 31) & 1)


def cond_holds(cpu, cond):
    n, z, c, v = cpu.n, cpu.z, cpu.c, cpu.v
    if cond == 0:
        return z
    if cond == 1:
        return not z
    if cond == 2:
        return c
    if cond == 3:
        return not c
    if cond == 4:
        return n
    if cond == 5:
        return not n
    if cond == 6:
        return v
    if cond == 7:
        return not v
    if cond == 8:
        return c and not z
    if cond == 9:
        return (not c) or z
    if cond == 10:
        return n == v
    if cond == 11:
        return n != v
    if cond == 12:
        return (not z) and (n == v)
    if cond == 13:
        return z or (n != v)
    return True


class UndefinedInstruction(Exception):
    def __init__(self, halfword, pc):
        super().__init__("undefined instruction 0x%04x at 0x%08x" % (halfword, pc))
        self.halfword = halfword
        self.pc = pc


def step(cpu):
    """Execute one instruction. Returns False once the core has halted."""
    if cpu.halted:
        return False

    # SysTick counts first so an interrupt is taken between instructions.
    if cpu.systick.step() and not cpu.primask:
        cpu.take_exception(SYSTICK_EXCEPTION)
        cpu.systick.pending = False
        if cpu.halted:
            return False

    pc = cpu.r[PC]
    op = cpu.mem.read16(pc)
    cpu.r[PC] = (pc + 2) & MASK32
    cpu.instructions += 1

    top = op >> 11

    # 000xx: shift by immediate (and 00011: add/sub register or 3-bit imm)
    if (op >> 13) == 0b000:
        if top != 0b00011:
            rd = op & 7
            rm = (op >> 3) & 7
            imm = (op >> 6) & 31
            kind = (op >> 11) & 3
            val = cpu.get(rm)
            if kind == 0:
                res, carry = lsl_c(val, imm)
            elif kind == 1:
                res, carry = lsr_c(val, 32 if imm == 0 else imm)
            else:
                res, carry = asr_c(val, 32 if imm == 0 else imm)
            cpu.set(rd, res)
            cpu.set_nz(res)
            if carry is not None:
                cpu.c = carry
            return True
        rd = op & 7
        rn = (op >> 3) & 7
        operand = (op >> 6) & 7
        is_imm = bool((op >> 10) & 1)
        is_sub = bool((op >> 9) & 1)
        b = operand if is_imm else cpu.get(operand)
        a = cpu.get(rn)
        if is_sub:
            res, carry, overflow = add_with_carry(a, (~b) & MASK32, True)
        else:
            res, carry, overflow = add_with_carry(a, b, False)
        cpu.set(rd, res)
        cpu.set_nz(res)
        cpu.c = carry
        cpu.v = overflow
        return True

    # 001xx: mov/cmp/add/sub 8-bit immediate
    if (op >> 13) == 0b001:
        rd = (op >> 8) & 7
        imm = op & 0xFF
        kind = (op >> 11) & 3
        if kind == 0:      # MOVS
            cpu.set(rd, imm)
            cpu.set_nz(imm)
        elif kind == 1:    # CMP
            res, carry, overflow = add_with_carry(cpu.get(rd), (~imm) & MASK32, True)
            cpu.set_nz(res)
            cpu.c = carry
            cpu.v = overflow
        elif kind == 2:    # ADDS
            res, carry, overflow = add_with_carry(cpu.get(rd), imm, False)
            cpu.set(rd, res)
            cpu.set_nz(res)
            cpu.c = carry
            cpu.v = overflow
        else:              # SUBS
            res, carry, overflow = add_with_carry(cpu.get(rd), (~imm) & MASK32, True)
            cpu.set(rd, res)
            cpu.set_nz(res)
            cpu.c = carry
            cpu.v = overflow
        return True

    # 010000: data processing register
    if (op >> 10) == 0b010000:
        code = (op >> 6) & 0xF
        rd = op & 7
        rm = (op >> 3) & 7
        a = cpu.get(rd)
        b = cpu.get(rm)
        if code == 0x0:    # ANDS
            res = a & b
            cpu.set(rd, res); cpu.set_nz(res)
        elif code == 0x1:  # EORS
            res = a ^ b
            cpu.set(rd, res); cpu.set_nz(res)
        elif code == 0x2:  # LSLS reg
            res, carry = lsl_c(a, b & 0xFF)
            cpu.set(rd, res); cpu.set_nz(res)
            if carry is not None:
                cpu.c = carry
        elif code == 0x3:  # LSRS reg
            res, carry = lsr_c(a, b & 0xFF)
            cpu.set(rd, res); cpu.set_nz(res)
            if carry is not None:
                cpu.c = carry
        elif code == 0x4:  # ASRS reg
            res, carry = asr_c(a, b & 0xFF)
            cpu.set(rd, res); cpu.set_nz(res)
            if carry is not None:
                cpu.c = carry
        elif code == 0x5:  # ADCS
            res, carry, overflow = add_with_carry(a, b, cpu.c)
            cpu.set(rd, res); cpu.set_nz(res); cpu.c = carry; cpu.v = overflow
        elif code == 0x6:  # SBCS
            res, carry, overflow = add_with_carry(a, (~b) & MASK32, cpu.c)
            cpu.set(rd, res); cpu.set_nz(res); cpu.c = carry; cpu.v = overflow
        elif code == 0x7:  # RORS
            res, carry = ror_c(a, b & 0xFF)
            cpu.set(rd, res); cpu.set_nz(res)
            if carry is not None:
                cpu.c = carry
        elif code == 0x8:  # TST
            cpu.set_nz(a & b)
        elif code == 0x9:  # RSBS (negate)
            res, carry, overflow = add_with_carry((~b) & MASK32, 0, True)
            cpu.set(rd, res); cpu.set_nz(res); cpu.c = carry; cpu.v = overflow
        elif code == 0xA:  # CMP
            res, carry, overflow = add_with_carry(a, (~b) & MASK32, True)
            cpu.set_nz(res); cpu.c = carry; cpu.v = overflow
        elif code == 0xB:  # CMN
            res, carry, overflow = add_with_carry(a, b, False)
            cpu.set_nz(res); cpu.c = carry; cpu.v = overflow
        elif code == 0xC:  # ORRS
            res = a | b
            cpu.set(rd, res); cpu.set_nz(res)
        elif code == 0xD:  # MULS
            res = (a * b) & MASK32
            cpu.set(rd, res); cpu.set_nz(res)
        elif code == 0xE:  # BICS
            res = a & (~b & MASK32)
            cpu.set(rd, res); cpu.set_nz(res)
        else:              # MVNS
            res = (~b) & MASK32
            cpu.set(rd, res); cpu.set_nz(res)
        return True

    # 010001: high register operations and BX
    if (op >> 10) == 0b010001:
        code = (op >> 8) & 3
        h1 = (op >> 7) & 1
        rm = (op >> 3) & 0xF
        rd = (op & 7) | (h1 << 3)
        if code == 0:      # ADD (no flags)
            val = (cpu.get(rd) + cpu.get(rm)) & MASK32
            if rd == PC:
                cpu.r[PC] = val & ~1
            else:
                cpu.set(rd, val)
        elif code == 1:    # CMP (flags only)
            res, carry, overflow = add_with_carry(cpu.get(rd), (~cpu.get(rm)) & MASK32, True)
            cpu.set_nz(res); cpu.c = carry; cpu.v = overflow
        elif code == 2:    # MOV (no flags)
            val = cpu.get(rm)
            if rd == PC:
                cpu.r[PC] = val & ~1
            else:
                cpu.set(rd, val)
        else:              # BX / BLX
            target = cpu.get(rm)
            if (op >> 7) & 1:  # BLX
                cpu.r[LR] = (cpu.r[PC]) | 1
            if target >= 0xFFFFFFF0:
                cpu.exception_return()
            else:
                cpu.r[PC] = target & ~1
        return True

    # 01001: LDR literal (PC-relative)
    if top == 0b01001:
        rd = (op >> 8) & 7
        imm = (op & 0xFF) << 2
        base = (cpu.get(PC)) & ~3
        cpu.set(rd, cpu.mem.read32(base + imm))
        return True

    # 0101: load/store with register offset
    if (op >> 12) == 0b0101:
        code = (op >> 9) & 7
        rd = op & 7
        rn = (op >> 3) & 7
        rm = (op >> 6) & 7
        addr = (cpu.get(rn) + cpu.get(rm)) & MASK32
        if code == 0:
            cpu.mem.write32(addr, cpu.get(rd))
        elif code == 1:
            cpu.mem.write16(addr, cpu.get(rd))
        elif code == 2:
            cpu.mem.write8(addr, cpu.get(rd))
        elif code == 3:
            val = cpu.mem.read8(addr)
            cpu.set(rd, val | (0xFFFFFF00 if val & 0x80 else 0))
        elif code == 4:
            cpu.set(rd, cpu.mem.read32(addr))
        elif code == 5:
            cpu.set(rd, cpu.mem.read16(addr))
        elif code == 6:
            cpu.set(rd, cpu.mem.read8(addr))
        else:
            val = cpu.mem.read16(addr)
            cpu.set(rd, val | (0xFFFF0000 if val & 0x8000 else 0))
        return True

    # 011: load/store word or byte, immediate offset
    if (op >> 13) == 0b011:
        is_byte = bool((op >> 12) & 1)
        is_load = bool((op >> 11) & 1)
        imm = (op >> 6) & 31
        rn = (op >> 3) & 7
        rd = op & 7
        addr = (cpu.get(rn) + (imm if is_byte else imm * 4)) & MASK32
        if is_load:
            cpu.set(rd, cpu.mem.read8(addr) if is_byte else cpu.mem.read32(addr))
        else:
            if is_byte:
                cpu.mem.write8(addr, cpu.get(rd))
            else:
                cpu.mem.write32(addr, cpu.get(rd))
        return True

    # 1000: load/store halfword immediate
    if (op >> 12) == 0b1000:
        is_load = bool((op >> 11) & 1)
        imm = ((op >> 6) & 31) * 2
        rn = (op >> 3) & 7
        rd = op & 7
        addr = (cpu.get(rn) + imm) & MASK32
        if is_load:
            cpu.set(rd, cpu.mem.read16(addr))
        else:
            cpu.mem.write16(addr, cpu.get(rd))
        return True

    # 1001: SP-relative load/store
    if (op >> 12) == 0b1001:
        is_load = bool((op >> 11) & 1)
        rd = (op >> 8) & 7
        imm = (op & 0xFF) * 4
        addr = (cpu.r[SP] + imm) & MASK32
        if is_load:
            cpu.set(rd, cpu.mem.read32(addr))
        else:
            cpu.mem.write32(addr, cpu.get(rd))
        return True

    # 1010: ADR / ADD Rd, SP, #imm
    if (op >> 12) == 0b1010:
        rd = (op >> 8) & 7
        imm = (op & 0xFF) * 4
        if (op >> 11) & 1:
            cpu.set(rd, (cpu.r[SP] + imm) & MASK32)
        else:
            cpu.set(rd, ((cpu.get(PC) & ~3) + imm) & MASK32)
        return True

    # 1011 0000: ADD/SUB SP, #imm7*4
    if (op >> 8) == 0b10110000:
        imm = (op & 0x7F) * 4
        if (op >> 7) & 1:
            cpu.r[SP] = (cpu.r[SP] - imm) & MASK32
        else:
            cpu.r[SP] = (cpu.r[SP] + imm) & MASK32
        return True

    # 1011 x10x: PUSH / POP
    if (op >> 12) == 0b1011 and ((op >> 9) & 3) == 0b10:
        is_pop = bool((op >> 11) & 1)
        extra = bool((op >> 8) & 1)
        regs = [i for i in range(8) if (op >> i) & 1]
        if is_pop:
            sp = cpu.r[SP]
            for i in regs:
                cpu.set(i, cpu.mem.read32(sp))
                sp += 4
            if extra:
                target = cpu.mem.read32(sp)
                sp += 4
                cpu.r[SP] = sp
                if target >= 0xFFFFFFF0:
                    cpu.exception_return()
                else:
                    cpu.r[PC] = target & ~1
                return True
            cpu.r[SP] = sp
        else:
            full = regs + ([LR] if extra else [])
            sp = cpu.r[SP] - 4 * len(full)
            cpu.r[SP] = sp
            for idx, i in enumerate(full):
                cpu.mem.write32(sp + idx * 4, cpu.r[i] if i == LR else cpu.get(i))
        return True

    # 1011 0110 011x: CPSIE i / CPSID i
    if (op & 0xFFE0) == 0xB660:
        cpu.primask = bool((op >> 4) & 1)
        return True

    # 1101: conditional branch and SVC
    if (op >> 12) == 0b1101:
        cond = (op >> 8) & 0xF
        if cond == 0xF:
            cpu.halted = True
            cpu.halt_reason = "svc 0x%02x" % (op & 0xFF)
            return False
        if cond == 0xE:
            raise UndefinedInstruction(op, pc)
        offset = op & 0xFF
        if offset & 0x80:
            offset -= 0x100
        if cond_holds(cpu, cond):
            cpu.r[PC] = (cpu.get(PC) + offset * 2) & MASK32
        return True

    # 11100: unconditional branch
    if top == 0b11100:
        offset = op & 0x7FF
        if offset & 0x400:
            offset -= 0x800
        target = (cpu.get(PC) + offset * 2) & MASK32
        if target == pc:
            cpu.halted = True
            cpu.halt_reason = "branch to self"
            return False
        cpu.r[PC] = target
        return True

    # 11110 / 11111: 32-bit BL
    if top == 0b11110:
        second = cpu.mem.read16(cpu.r[PC])
        cpu.r[PC] = (cpu.r[PC] + 2) & MASK32
        s = (op >> 10) & 1
        imm10 = op & 0x3FF
        j1 = (second >> 13) & 1
        j2 = (second >> 11) & 1
        imm11 = second & 0x7FF
        i1 = 1 - (j1 ^ s)
        i2 = 1 - (j2 ^ s)
        offset = (s << 24) | (i1 << 23) | (i2 << 22) | (imm10 << 12) | (imm11 << 1)
        if s:
            offset -= 1 << 25
        cpu.r[LR] = (cpu.r[PC]) | 1
        cpu.r[PC] = ((pc + 4) + offset) & MASK32
        return True

    raise UndefinedInstruction(op, pc)


def run(cpu, max_instructions=1000000):
    """Run until halt or the instruction budget is spent."""
    while cpu.instructions < max_instructions:
        if not step(cpu):
            return cpu.halt_reason or "halted"
    return "instruction budget exhausted"

