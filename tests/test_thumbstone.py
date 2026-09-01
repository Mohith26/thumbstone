"""Test suite. Three layers, in increasing order of how much they prove:

1. Flag semantics checked against an independent oracle written with Python
   big integers, swept over adversarial operand pairs. If the emulator and
   the oracle disagree on N/Z/C/V for any pair, the test names the pair.
2. Whole programs assembled from source, run to completion, and compared
   against results computed in Python.
3. Exception behaviour: SysTick entry, register preservation, masking.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thumbstone.cpu import CPU, Memory, MASK32, SP, LR, PC, SYSTICK_EXCEPTION, MemoryFault
from thumbstone.execute import (step, run, add_with_carry, lsl_c, lsr_c, asr_c,
                                sign32, UndefinedInstruction)
from thumbstone.assembler import assemble, AsmError

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append(name)
        print("  FAIL %s %s" % (name, detail))


def check_eq(name, got, want):
    check(name, got == want, "got %r want %r" % (got, want))


# ---------------------------------------------------------------- oracle ---

INTERESTING = [0, 1, 2, 7, 0x7F, 0x80, 0xFF, 0x100, 0x7FFF, 0x8000,
               0xFFFF, 0x10000, 0x7FFFFFFE, 0x7FFFFFFF, 0x80000000,
               0x80000001, 0xFFFFFFFE, 0xFFFFFFFF, 0x5A5A5A5A, 0xA5A5A5A5,
               0xDEADBEEF, 0xCAFEBABE, 3, 0xFFFFFFFD]


def oracle_add(a, b, carry_in):
    """Flags for a + b + carry, computed independently with big ints."""
    usum = a + b + carry_in
    result = usum & MASK32
    carry = usum > MASK32
    ssum = sign32(a) + sign32(b) + carry_in
    overflow = not (-(1 << 31) <= ssum <= (1 << 31) - 1)
    n = bool(result >> 31)
    z = result == 0
    return result, n, z, carry, overflow


def test_add_with_carry_matches_oracle():
    bad = []
    for a in INTERESTING:
        for b in INTERESTING:
            for cin in (0, 1):
                res, carry, ov = add_with_carry(a, b, bool(cin))
                ores, on, oz, ocarry, oov = oracle_add(a, b, cin)
                if (res, carry, ov) != (ores, ocarry, oov):
                    bad.append((hex(a), hex(b), cin, (res, carry, ov), (ores, ocarry, oov)))
    check("add_with_carry matches the oracle over %d pairs" % (len(INTERESTING) ** 2 * 2),
          not bad, str(bad[:3]))


def test_subtraction_carry_convention():
    # SUB is a + ~b + 1. Carry set means no borrow, which is the opposite of
    # what most people guess, so pin it down explicitly.
    res, carry, ov = add_with_carry(5, (~3) & MASK32, True)
    check_eq("5 - 3 = 2", res, 2)
    check("5 - 3 sets carry (no borrow)", carry)
    res, carry, ov = add_with_carry(3, (~5) & MASK32, True)
    check_eq("3 - 5 wraps", res, (3 - 5) & MASK32)
    check("3 - 5 clears carry (borrow)", not carry)
    # signed overflow: most negative minus one
    res, carry, ov = add_with_carry(0x80000000, (~1) & MASK32, True)
    check("INT_MIN - 1 overflows", ov)


def test_shift_carry_edges():
    check_eq("lsl by 0 leaves carry alone", lsl_c(0xFFFFFFFF, 0)[1], None)
    check_eq("lsl by 1 carries out the top bit", lsl_c(0x80000000, 1), (0, True))
    check_eq("lsl by 32 is zero, carry from bit 0", lsl_c(1, 32), (0, True))
    check_eq("lsr by 32 is zero, carry from bit 31", lsr_c(0x80000000, 32), (0, True))
    check_eq("asr by 32 of a negative is all ones", asr_c(0x80000000, 32), (0xFFFFFFFF, True))
    check_eq("asr by 32 of a positive is zero", asr_c(0x7FFFFFFF, 32), (0, False))
    check_eq("asr keeps the sign", asr_c(0xFFFFFFF0, 4)[0], 0xFFFFFFFF)
    check_eq("lsr does not keep the sign", lsr_c(0xFFFFFFF0, 4)[0], 0x0FFFFFFF)


# ------------------------------------------------------------- assembler ---

def build(source, origin=0x100, stack=0x8000, mem_size=1 << 16):
    code, labels = assemble(source, origin)
    mem = Memory(mem_size)
    mem.load_image(origin, code)
    cpu = CPU(mem)
    cpu.r[PC] = origin
    cpu.r[SP] = stack
    return cpu, labels, code


def test_assembler_round_trip():
    code, labels = assemble("""
start:
  movs r0, #1
  movs r1, #2
  adds r2, r0, r1
  b .
""", 0x100)
    check_eq("assembler emits the right size", len(code), 8)
    check_eq("labels resolve to addresses", labels["start"], 0x100)
    check_eq("movs r0,#1 encodes as 0x2001", code[0] | (code[1] << 8), 0x2001)
    check_eq("adds r2,r0,r1 encodes as 0x1842", code[4] | (code[5] << 8), 0x1842)


def test_assembler_rejects_bad_input():
    for src, why in [
        ("movs r0, #300", "immediate out of range"),
        ("adds r0, r1, #9", "3-bit immediate"),
        ("ldr r0, [r1, #3]", "unaligned word offset"),
        ("push {r9}", "high register in push"),
        ("frobnicate r0", "unknown mnemonic"),
    ]:
        raised = False
        try:
            assemble(src, 0x100)
        except AsmError:
            raised = True
        check("assembler rejects: %s" % why, raised)


# -------------------------------------------------------------- programs ---

FIB = """
start:
  movs r0, #0
  movs r1, #1
  movs r2, #0
loop:
  cmp r2, r3
  bge done
  mov r4, r1
  adds r1, r0, r1
  mov r0, r4
  adds r2, r2, #1
  b loop
done:
  b .
"""


def test_fibonacci():
    for n in (0, 1, 2, 5, 10, 20, 30):
        cpu, labels, _ = build(FIB)
        cpu.set(3, n)
        run(cpu, 100000)
        expected = 0
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        expected = a & MASK32
        check_eq("fib(%d)" % n, cpu.get(0), expected)


MEMCPY = """
start:
  movs r4, #0
copy:
  cmp r4, r3
  beq done
  ldrb r5, [r0, #0]
  strb r5, [r1, #0]
  adds r0, r0, #1
  adds r1, r1, #1
  adds r4, r4, #1
  b copy
done:
  b .
"""


def test_memcpy():
    cpu, labels, _ = build(MEMCPY)
    src_addr, dst_addr, n = 0x2000, 0x3000, 64
    payload = bytes((i * 7 + 3) & 0xFF for i in range(n))
    cpu.mem.load_image(src_addr, payload)
    cpu.set(0, src_addr)
    cpu.set(1, dst_addr)
    cpu.set(3, n)
    run(cpu, 100000)
    copied = bytes(cpu.mem.buf[dst_addr:dst_addr + n])
    check_eq("memcpy moves the bytes exactly", copied, payload)
    check("memcpy did not touch the byte after the range", cpu.mem.read8(dst_addr + n) == 0)


BUBBLE = """
start:
outer:
  movs r4, #0
  movs r5, #0
inner:
  adds r6, r5, #1
  cmp r6, r1
  bge endinner
  lsls r7, r5, #2
  adds r7, r0, r7
  ldr r2, [r7, #0]
  ldr r3, [r7, #4]
  cmp r2, r3
  ble noswap
  str r3, [r7, #0]
  str r2, [r7, #4]
  movs r4, #1
noswap:
  adds r5, r5, #1
  b inner
endinner:
  cmp r4, #0
  bne outer
  b .
"""


def test_bubble_sort():
    cpu, labels, _ = build(BUBBLE)
    values = [42, 7, 19, 3, 88, 1, 64, 25, 13, 99, 2, 56]
    base = 0x4000
    for i, v in enumerate(values):
        cpu.mem.write32(base + i * 4, v)
    cpu.set(0, base)
    cpu.set(1, len(values))
    run(cpu, 500000)
    got = [cpu.mem.read32(base + i * 4) for i in range(len(values))]
    check_eq("bubble sort orders the array", got, sorted(values))


CALL = """
start:
  movs r0, #6
  bl square
  mov r5, r0
  movs r0, #9
  bl square
  adds r5, r5, r0
  b .
square:
  push {r4, lr}
  mov r4, r0
  muls r0, r4
  pop {r4, pc}
"""


def test_bl_and_return():
    cpu, labels, _ = build(CALL)
    run(cpu, 10000)
    check_eq("bl/pop-pc call and return", cpu.get(5), 36 + 81)
    check_eq("stack is balanced afterwards", cpu.r[SP], 0x8000)


def test_push_pop_preserves_registers():
    cpu, labels, _ = build("""
start:
  movs r1, #11
  movs r2, #22
  push {r1, r2}
  movs r1, #99
  movs r2, #99
  pop {r1, r2}
  b .
""")
    run(cpu, 1000)
    check_eq("pop restores r1", cpu.get(1), 11)
    check_eq("pop restores r2", cpu.get(2), 22)
    check_eq("sp returns to where it started", cpu.r[SP], 0x8000)


# ------------------------------------------------------------ exceptions ---

TICKER = """
start:
  movs r0, #0
  cpsie
wait:
  adds r0, r0, #1
  b wait
handler:
  push {r1, lr}
  movs r1, #1
  adds r6, r6, r1
  pop {r1, pc}
"""


def drain_to_thread_mode(cpu, limit=200):
    """Stop the timer and let any in-flight handler finish.

    Without this the instruction budget can expire midway through a handler,
    and the test then compares a half-finished exception against a completed
    one. That is a measurement artefact, not a core bug, so the test removes
    it rather than loosening the assertion.
    """
    cpu.systick.enabled = False
    steps = 0
    while cpu.in_handler and steps < limit and not cpu.halted:
        step(cpu)
        steps += 1
    return steps


def test_systick_interrupt():
    cpu, labels, _ = build(TICKER)
    cpu.vectors[SYSTICK_EXCEPTION] = labels["handler"]
    cpu.systick.configure(50)
    run(cpu, 5000)
    drain_to_thread_mode(cpu)
    check("systick fired repeatedly", cpu.systick.count > 5)
    check_eq("handler ran once per tick", cpu.get(6), cpu.systick.count)
    check_eq("returned to thread mode", cpu.in_handler, False)
    check_eq("stack balanced after all the exceptions", cpu.r[SP], 0x8000)


def test_exception_preserves_caller_registers():
    cpu, labels, _ = build("""
start:
  movs r0, #7
  movs r1, #8
  movs r2, #9
  movs r3, #10
  cpsie
spin:
  b spin
handler:
  movs r0, #99
  movs r1, #99
  movs r2, #99
  movs r3, #99
  bx lr
""")
    cpu.vectors[SYSTICK_EXCEPTION] = labels["handler"]
    cpu.systick.configure(20)
    run(cpu, 400)
    check_eq("r0 restored by exception return", cpu.get(0), 7)
    check_eq("r1 restored", cpu.get(1), 8)
    check_eq("r2 restored", cpu.get(2), 9)
    check_eq("r3 restored", cpu.get(3), 10)


def test_primask_blocks_interrupts():
    cpu, labels, _ = build("""
start:
  cpsid
spin:
  adds r6, r6, #1
  b spin
handler:
  adds r7, r7, #1
  bx lr
""")
    cpu.vectors[SYSTICK_EXCEPTION] = labels["handler"]
    cpu.systick.configure(10)
    run(cpu, 500)
    check_eq("masked interrupts never dispatch", cpu.get(7), 0)
    check("the main loop still ran", cpu.get(6) > 100)


# ----------------------------------------------------------------- traps ---

def test_memory_faults():
    mem = Memory(1024)
    for addr, width, what in [(1, 4, "unaligned word"), (2, 4, "unaligned word"),
                              (1, 2, "unaligned halfword"), (1020, 8, "past the end")]:
        raised = False
        try:
            if width == 4:
                mem.read32(addr)
            elif width == 2:
                mem.read16(addr)
            else:
                mem.read32(addr + 4)
        except MemoryFault:
            raised = True
        check("memory fault on %s" % what, raised)


def test_undefined_instruction():
    mem = Memory(1024)
    mem.write16(0x100, 0xDE00)  # permanently undefined encoding
    cpu = CPU(mem)
    cpu.r[PC] = 0x100
    cpu.r[SP] = 0x300
    raised = False
    try:
        step(cpu)
    except UndefinedInstruction:
        raised = True
    check("undefined instruction raises", raised)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print("thumbstone test suite (%d test functions)" % len(tests))
    for fn in tests:
        fn()
    print("\n%d checks passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return len(FAIL)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)

