"""Throughput and coverage numbers, written to bench-results.json.

Instruction rate here is a property of this interpreter, not of any real
part. It is reported because it bounds how big a workload you can simulate
in a test loop, which is the only reason the number matters.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from thumbstone.cpu import CPU, Memory, MASK32, SP, PC, SYSTICK_EXCEPTION
from thumbstone.execute import run, add_with_carry, sign32
from thumbstone.assembler import assemble

SUM_LOOP = """
start:
  movs r0, #0
  movs r1, #0
loop:
  adds r0, r0, r1
  adds r1, r1, #1
  cmp r1, r2
  bne loop
  b .
"""

SORT = """
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


def build(source, origin=0x100, stack=0x8000, size=1 << 17):
    code, labels = assemble(source, origin)
    mem = Memory(size)
    mem.load_image(origin, code)
    cpu = CPU(mem)
    cpu.r[PC] = origin
    cpu.r[SP] = stack
    return cpu, labels


def bench_loop():
    cpu, _ = build(SUM_LOOP)
    cpu.set(2, 20000)
    t0 = time.time()
    run(cpu, 5000000)
    elapsed = time.time() - t0
    expected = sum(range(20000)) & MASK32
    return {
        "program": "accumulate 0..19999",
        "instructions": cpu.instructions,
        "seconds": round(elapsed, 4),
        "instructionsPerSecond": int(cpu.instructions / elapsed) if elapsed else None,
        "result": cpu.get(0),
        "expected": expected,
        "correct": cpu.get(0) == expected,
    }


def bench_sort(n=120):
    cpu, _ = build(SORT)
    base = 0x9000
    values = [(i * 7919 + 13) % 100000 for i in range(n)]
    for i, v in enumerate(values):
        cpu.mem.write32(base + i * 4, v)
    cpu.set(0, base)
    cpu.set(1, n)
    t0 = time.time()
    run(cpu, 20000000)
    elapsed = time.time() - t0
    got = [cpu.mem.read32(base + i * 4) for i in range(n)]
    return {
        "program": "bubble sort %d words" % n,
        "instructions": cpu.instructions,
        "seconds": round(elapsed, 4),
        "instructionsPerSecond": int(cpu.instructions / elapsed) if elapsed else None,
        "sortedCorrectly": got == sorted(values),
    }


def bench_interrupts():
    cpu, labels = build("""
start:
  cpsie
spin:
  adds r0, r0, #1
  b spin
handler:
  push {r1, lr}
  adds r1, r6, #1
  mov r6, r1
  pop {r1, pc}
""")
    cpu.vectors[SYSTICK_EXCEPTION] = labels["handler"]
    cpu.systick.configure(25)
    t0 = time.time()
    run(cpu, 200000)
    elapsed = time.time() - t0
    return {
        "program": "systick storm, reload 25",
        "instructions": cpu.instructions,
        "seconds": round(elapsed, 4),
        "exceptionsTaken": cpu.exceptions_taken,
        "handlerRuns": cpu.get(6),
        "exceptionsPerSecond": int(cpu.exceptions_taken / elapsed) if elapsed else None,
    }


def bench_flag_oracle():
    values = [0, 1, 2, 0x7F, 0x80, 0xFF, 0x7FFF, 0x8000, 0xFFFF,
              0x7FFFFFFE, 0x7FFFFFFF, 0x80000000, 0x80000001,
              0xFFFFFFFE, 0xFFFFFFFF, 0x5A5A5A5A, 0xA5A5A5A5, 3]
    mismatches = 0
    checked = 0
    t0 = time.time()
    for a in values:
        for b in values:
            for cin in (0, 1):
                checked += 1
                res, carry, ov = add_with_carry(a, b, bool(cin))
                usum = a + b + cin
                oref = usum & MASK32
                ocarry = usum > MASK32
                ssum = sign32(a) + sign32(b) + cin
                oov = not (-(1 << 31) <= ssum <= (1 << 31) - 1)
                if (res, carry, ov) != (oref, ocarry, oov):
                    mismatches += 1
    return {
        "combinationsChecked": checked,
        "mismatches": mismatches,
        "seconds": round(time.time() - t0, 4),
    }


def main():
    out = {
        "note": "run under CPython; instruction rates are interpreter-bound, not silicon estimates",
        "sumLoop": bench_loop(),
        "bubbleSort": bench_sort(),
        "interrupts": bench_interrupts(),
        "flagOracle": bench_flag_oracle(),
    }
    print(json.dumps(out, indent=2))
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "bench-results.json"), "w") as fh:
        fh.write(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    main()

