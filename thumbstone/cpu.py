"""Core machine state: registers, flags, memory and the exception model.

Everything is 32-bit little-endian and masked on every write, because the
single most common source of wrong answers in a hand-written emulator is a
Python integer quietly growing past 32 bits and making a comparison come out
right when the real silicon would have wrapped.
"""

MASK32 = 0xFFFFFFFF

SP = 13
LR = 14
PC = 15

# Values the core writes into LR on exception entry. Only the main-stack
# variants are modelled here; this core has no process stack.
EXC_RETURN_HANDLER = 0xFFFFFFF1
EXC_RETURN_THREAD = 0xFFFFFFF9

SYSTICK_EXCEPTION = 15


class MemoryFault(Exception):
    def __init__(self, addr, size, what):
        super().__init__("%s at 0x%08x (size %d)" % (what, addr, size))
        self.addr = addr
        self.size = size


class Memory:
    """Flat byte-addressed memory with alignment checks.

    ARMv6-M does not support unaligned word or halfword access, so an
    unaligned load is a fault rather than something quietly fixed up. Getting
    this wrong makes buggy firmware look like it works.
    """

    def __init__(self, size=1 << 16):
        self.size = size
        self.buf = bytearray(size)

    def _check(self, addr, width):
        if addr < 0 or addr + width > self.size:
            raise MemoryFault(addr, width, "out of range")
        if width > 1 and (addr % width) != 0:
            raise MemoryFault(addr, width, "unaligned access")

    def read8(self, addr):
        self._check(addr, 1)
        return self.buf[addr]

    def read16(self, addr):
        self._check(addr, 2)
        return self.buf[addr] | (self.buf[addr + 1] << 8)

    def read32(self, addr):
        self._check(addr, 4)
        b = self.buf
        return b[addr] | (b[addr + 1] << 8) | (b[addr + 2] << 16) | (b[addr + 3] << 24)

    def write8(self, addr, val):
        self._check(addr, 1)
        self.buf[addr] = val & 0xFF

    def write16(self, addr, val):
        self._check(addr, 2)
        self.buf[addr] = val & 0xFF
        self.buf[addr + 1] = (val >> 8) & 0xFF

    def write32(self, addr, val):
        self._check(addr, 4)
        v = val & MASK32
        self.buf[addr] = v & 0xFF
        self.buf[addr + 1] = (v >> 8) & 0xFF
        self.buf[addr + 2] = (v >> 16) & 0xFF
        self.buf[addr + 3] = (v >> 24) & 0xFF

    def load_image(self, addr, data):
        self._check(addr, 1)
        if addr + len(data) > self.size:
            raise MemoryFault(addr, len(data), "image does not fit")
        self.buf[addr:addr + len(data)] = data


class SysTick:
    """The 24-bit down counter every Cortex-M part has.

    Counts down once per instruction here rather than once per clock: this
    core does not model cycle timing, and pretending otherwise would put a
    fake number in the output.
    """

    def __init__(self):
        self.enabled = False
        self.reload = 0
        self.current = 0
        self.tick_int = False
        self.pending = False
        self.count = 0

    def configure(self, reload_value, interrupt=True):
        self.reload = reload_value & 0xFFFFFF
        self.current = self.reload
        self.tick_int = interrupt
        self.enabled = True

    def step(self):
        if not self.enabled or self.reload == 0:
            return False
        if self.current == 0:
            self.current = self.reload
        self.current -= 1
        if self.current == 0:
            self.count += 1
            if self.tick_int:
                self.pending = True
                return True
        return False


class CPU:
    def __init__(self, memory=None):
        self.mem = memory if memory is not None else Memory()
        self.r = [0] * 16
        self.n = False
        self.z = False
        self.c = False
        self.v = False
        self.primask = False          # interrupts masked (CPSID i)
        self.systick = SysTick()
        self.vectors = {}             # exception number -> handler address
        self.in_handler = False
        self.instructions = 0
        self.exceptions_taken = 0
        self.halted = False
        self.halt_reason = None

    # register access ------------------------------------------------------
    def get(self, i):
        if i == PC:
            # Reading PC yields the address of the *current* instruction plus
            # 4, which is the pipeline behaviour PC-relative loads and
            # branches depend on. The decoder has already advanced r[PC] past
            # the halfword it fetched, so r[PC] is current+2 and the offset
            # to add here is 2, not 4. Adding 4 here double-counts the fetch
            # and lands every PC-relative address two bytes high, which is
            # subtle enough that literal loads still return plausible data.
            return (self.r[PC] + 2) & MASK32
        return self.r[i]

    def set(self, i, val):
        self.r[i] = val & MASK32

    def set_nz(self, val):
        v = val & MASK32
        self.n = (v >> 31) == 1
        self.z = v == 0

    def flags_tuple(self):
        return (self.n, self.z, self.c, self.v)

    def load_flags(self, xpsr):
        self.n = bool(xpsr & (1 << 31))
        self.z = bool(xpsr & (1 << 30))
        self.c = bool(xpsr & (1 << 29))
        self.v = bool(xpsr & (1 << 28))

    def xpsr(self):
        val = 0
        if self.n:
            val |= 1 << 31
        if self.z:
            val |= 1 << 30
        if self.c:
            val |= 1 << 29
        if self.v:
            val |= 1 << 28
        val |= 1 << 24  # Thumb bit, always set on this profile
        return val

    # exceptions -----------------------------------------------------------
    def take_exception(self, number):
        """Hardware stacking, exactly the eight words the core pushes.

        Order from low address up: R0 R1 R2 R3 R12 LR ReturnAddress xPSR.
        """
        handler = self.vectors.get(number)
        if handler is None:
            self.halted = True
            self.halt_reason = "no handler for exception %d" % number
            return
        sp = self.r[SP]
        frame = [self.r[0], self.r[1], self.r[2], self.r[3],
                 self.r[12], self.r[LR], self.r[PC], self.xpsr()]
        sp -= 32
        if sp < 0:
            raise MemoryFault(sp, 32, "stack underflow during exception entry")
        for idx, word in enumerate(frame):
            self.mem.write32(sp + idx * 4, word)
        self.r[SP] = sp
        self.r[LR] = EXC_RETURN_HANDLER if self.in_handler else EXC_RETURN_THREAD
        self.in_handler = True
        self.r[PC] = handler & ~1
        self.exceptions_taken += 1

    def exception_return(self):
        sp = self.r[SP]
        self.r[0] = self.mem.read32(sp)
        self.r[1] = self.mem.read32(sp + 4)
        self.r[2] = self.mem.read32(sp + 8)
        self.r[3] = self.mem.read32(sp + 12)
        self.r[12] = self.mem.read32(sp + 16)
        self.r[LR] = self.mem.read32(sp + 20)
        self.r[PC] = self.mem.read32(sp + 24) & ~1
        self.load_flags(self.mem.read32(sp + 28))
        self.r[SP] = sp + 32
        self.in_handler = False

    def dump(self):
        return {
            "r": list(self.r),
            "flags": {"n": self.n, "z": self.z, "c": self.c, "v": self.v},
            "instructions": self.instructions,
            "exceptions": self.exceptions_taken,
            "systick_count": self.systick.count,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }

