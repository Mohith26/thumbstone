# thumbstone

An ARMv6-M (Cortex-M0 class) instruction set simulator written in plain
Python, with a small assembler so the test programs read as assembly instead
of hex, and an exception model good enough to run interrupt-driven code.

I wanted to actually understand the machine underneath embedded C rather
than take the compiler's word for it. Writing the decoder is the easy half;
the half that teaches you something is getting the flags right, because
almost every wrong answer traces back to a carry or overflow rule you
assumed instead of looked up.

## What it runs

The 16-bit Thumb subset an ARMv6-M part actually uses, plus 32-bit `BL`:

- shifts by immediate (`LSLS`/`LSRS`/`ASRS`), including the encoding quirk
  where a shift amount of zero means 32 for LSR and ASR
- add/subtract, register and 3-bit immediate forms
- the 8-bit immediate group (`MOVS`, `CMP`, `ADDS`, `SUBS`)
- all sixteen data-processing register operations, `ANDS` through `MVNS`
- high-register `ADD`/`CMP`/`MOV` and `BX`, including `BX LR` from a handler
- loads and stores: literal (PC-relative), register offset, immediate
  offset, byte, halfword, sign-extending byte and halfword, SP-relative
- `ADR`, `ADD/SUB SP`, `PUSH`/`POP` with LR/PC
- conditional branches over all fourteen usable conditions, `B`, `BL`
- `CPSIE i` / `CPSID i`, and `SVC` (used here as a clean halt)

Memory is little-endian with real alignment checking: an unaligned word or
halfword access faults instead of being quietly fixed up, because on this
profile that is what the hardware does.

Exceptions model SysTick: the eight-word hardware stack frame
(R0-R3, R12, LR, return address, xPSR), an `EXC_RETURN` value in LR, and
unstacking on `BX LR` or `POP {..., PC}`. PRIMASK masking works.

## Correctness, and how it is argued

Three layers, in increasing order of what they prove:

1. **Flags against an independent oracle.** `add_with_carry` is swept
   against a separately written big-integer reference over adversarial
   operand pairs (signed boundaries, all-ones, alternating bit patterns).
   The bench run checks 648 combinations with **0 mismatches**; the test
   suite sweeps 1,152. Subtraction is included specifically because the
   carry convention (C set means *no borrow*) is the one people get wrong.
2. **Whole programs.** Fibonacci across seven values, a byte-wise memcpy
   verified against the source buffer, a bubble sort of twelve words
   compared to Python's `sorted()`, and a nested call chain using
   `BL`/`PUSH {r4, lr}`/`POP {r4, pc}` with a stack-balance assertion.
3. **Exceptions.** A SysTick storm where the handler count must equal the
   tick count and the stack must return to where it started, a test that
   caller registers R0-R3 survive a handler that deliberately clobbers
   them, and a test that PRIMASK actually blocks dispatch.

53 checks, all passing.

## Measured

From `bench.py` (CPython; these rates are interpreter-bound and say nothing
about real silicon):

- 80,003 instructions of an accumulate loop at 776,728 instructions/sec,
  result verified against Python's own sum.
- 157,419 instructions of a 120-word bubble sort at 832,904
  instructions/sec, output verified against `sorted()`.
- A SysTick storm taking 8,000 exceptions in 200,000 instructions at
  26,229 exceptions/sec, with handler runs matching ticks.

## Three bugs worth writing down

**PC read offset.** Reading PC must yield the address of the *current*
instruction plus 4. The decoder advances `r[PC]` past the fetched halfword
before executing, so the correct addend at that point is 2, not 4. Adding 4
put every PC-relative address two bytes high. The tell was that literal
loads still returned plausible-looking data, so nothing crashed; what broke
was `b .`, whose self-branch halt check silently stopped matching, and a
test program ran off into a subroutine and kept going.

**Assembler argument splitting.** The splitter tracked braces (for
`{r4, lr}`) but not brackets, so `ldr r2, [r7, #0]` split down the middle
into three arguments and missed the two-argument load path entirely. Every
immediate-offset load and store failed to assemble.

**CPSID/CPSIE encodings swapped.** `CPSID i` is `0xB672` and `CPSIE i` is
`0xB662`; I had emitted `0xB662` for `CPSID`, so the instruction that was
supposed to mask interrupts enabled them instead. The PRIMASK test caught
it by observing a handler run 50 times when it should have run zero.

## Deliberate limits

- No cycle accounting. SysTick counts down once per instruction, not once
  per clock, so any "cycles" number would be invented. It is not reported.
- No MPU, no BASEPRI, no priority levels or nesting beyond a single active
  handler, no process stack pointer, no sleep modes.
- No ELF loading and no C toolchain integration; programs come from the
  bundled assembler or raw bytes.
- The assembler has no literal pools, so `ldr rd, =value` is not supported;
  use `.word` and a PC-relative load.

## Run it

```
python3 tests/test_thumbstone.py
python3 bench.py
```

Standard library only.

