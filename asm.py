#!/usr/bin/env python3
import sys
from exceptions import (
    OutOfMemoryError,
    InvalidFreeError,
    InvalidPointerError,
    DataTooLargeError,
)
import pytest

class MemoryAllocator:
    def __init__(self, size):
        self.memory = bytearray(size)
        self.free_list = [(0, size)]  # (start, length)
        self.allocations = {}         # ptr -> size

    def malloc(self, size):
        for i, (start, length) in enumerate(self.free_list):
            if length >= size:
                self.allocations[start] = size
                if length == size:
                    self.free_list.pop(i)
                else:
                    self.free_list[i] = (start + size, length - size)
                return start
        raise OutOfMemoryError("Out of memory!")

    def free(self, ptr):
        if ptr not in self.allocations:
            raise InvalidFreeError(f"Invalid free: {ptr}")
        size = self.allocations.pop(ptr)
        self.free_list.append((ptr, size))
        self._coalesce()

    def realloc(self, ptr, new_size):
        if ptr is None:
            return self.malloc(new_size)
        if new_size == 0:
            self.free(ptr)
            return None

        old_size = self.allocations.get(ptr)
        if old_size is None:
            raise InvalidPointerError(f"Invalid pointer: {ptr}")

        if new_size <= old_size:
            self.allocations[ptr] = new_size
            leftover = old_size - new_size
            if leftover > 0:
                self.free_list.append((ptr + new_size, leftover))
                self._coalesce()
            return ptr

        end = ptr + old_size
        for i, (start, length) in enumerate(self.free_list):
            if start == end and length >= (new_size - old_size):
                extra = new_size - old_size
                self.allocations[ptr] = new_size
                if length == extra:
                    self.free_list.pop(i)
                else:
                    self.free_list[i] = (start + extra, length - extra)
                return ptr

        new_ptr = self.malloc(new_size)
        self.memory[new_ptr:new_ptr+old_size] = self.memory[ptr:ptr+old_size]
        self.free(ptr)
        return new_ptr

    def _coalesce(self):
        self.free_list.sort()
        merged = []
        prev_start, prev_len = self.free_list[0]
        for start, length in self.free_list[1:]:
            if prev_start + prev_len == start:
                prev_len += length
            else:
                merged.append((prev_start, prev_len))
                prev_start, prev_len = start, length
        merged.append((prev_start, prev_len))
        self.free_list = merged

    def write(self, ptr, data: bytes):
        size = self.allocations.get(ptr)
        if size is None:
            raise InvalidPointerError(f"Invalid pointer: {ptr}")
        if len(data) > size:
            raise DataTooLargeError(
                f"Block size {size}, tried to write {len(data)} bytes"
            )
        self.memory[ptr:ptr+len(data)] = data

    def read(self, ptr):
        size = self.allocations.get(ptr)
        if size is None:
            raise InvalidPointerError(f"Invalid pointer: {ptr}")
        return bytes(self.memory[ptr:ptr+size])

def test_malloc_and_write_read():
    alloc = MemoryAllocator(64)

    # allocate 10 bytes
    p = alloc.malloc(10)
    assert p in alloc.allocations
    assert alloc.allocations[p] == 10

    # write and read
    alloc.write(p, b"hello")
    assert alloc.read(p)[:5] == b"hello"


def test_free_and_reuse():
    alloc = MemoryAllocator(32)
    p1 = alloc.malloc(8)
    alloc.write(p1, b"12345678")
    alloc.free(p1)

    # after free, malloc should reuse
    p2 = alloc.malloc(8)
    assert p1 == p2


def test_realloc_expand_and_shrink():
    alloc = MemoryAllocator(64)

    # allocate 10, write data
    p = alloc.malloc(10)
    alloc.write(p, b"abcde")

    # expand to 20
    p2 = alloc.realloc(p, 20)
    assert p2 in alloc.allocations
    assert alloc.read(p2)[:5] == b"abcde"

    # shrink to 5
    p3 = alloc.realloc(p2, 5)
    assert alloc.allocations[p3] == 5
    assert alloc.read(p3)[:5] == b"abcde"


def test_realloc_free_case():
    alloc = MemoryAllocator(32)
    p = alloc.malloc(8)
    p = alloc.realloc(p, 0)  # acts like free
    assert p is None
    assert alloc.allocations == {}


def test_out_of_memory():
    alloc = MemoryAllocator(16)
    alloc.malloc(16)
    with pytest.raises(OutOfMemoryError):
        alloc.malloc(1)


def test_invalid_free():
    alloc = MemoryAllocator(16)
    with pytest.raises(InvalidFreeError):
        alloc.free(5)


def test_invalid_pointer_read_write():
    alloc = MemoryAllocator(16)

    with pytest.raises(InvalidPointerError):
        alloc.read(5)

    p = alloc.malloc(8)
    with pytest.raises(DataTooLargeError):
        alloc.write(p, b"123456789")  # 9 > 8

SIZE = int(1024 * 1024 * (1/8))  # 128KB memory
memory = MemoryAllocator(SIZE)
vars = {"ax": 0, "bx": 0, "cx": 0, "dx": 0, "pstack": [], "stack": [], "astack": [], "cstack": [], "cmpf": 0, "pc": 0}  # registers and stacks
labels = {}
custom_var_addrs = {}
functions = {}
AX = "ax"
BX = "bx"
CX = "cx"
DX = "dx"
PSTACK = "pstack"
STACK = "stack"
ASTACK = "astack"
CSTACK = "cstack"
CMPF = "cmpf"
PC = "pc"
import re
def parse_address(expr: str) -> tuple[int, int]:
    """
    Parse memory reference like [ptr+8,16] or [0x1000-4,4].
    Returns (absolute_address, length).
    """
    if not (isinstance(expr, str) and expr.startswith("[") and expr.endswith("]")):
        raise InvalidPointerError(f"Invalid memory reference: {expr}")

    inside = expr[1:-1].strip()
    # support optional ,len
    if "," in inside:
        base_part, len_part = inside.split(",", 1)
        length = int(len_part.strip())
    else:
        base_part, length = inside, 8  # default = 8 bytes

    m = re.match(r"(.+?)([+-]\d+)?$", base_part.strip())
    if not m:
        raise InvalidPointerError(f"Invalid memory address: {expr}")

    base_str, offset_str = m.groups()
    base_str = base_str.strip()

    # resolve base
    if base_str in vars:
        base = vars[base_str]
    elif base_str in custom_var_addrs:
        base = custom_var_addrs[base_str]
    else:
        try:
            base = to_int(base_str)
        except Exception:
            raise InvalidPointerError(f"Invalid base in memory reference: {base_str}")

    offset = int(offset_str) if offset_str else 0
    addr = base + offset

    if addr < 0 or addr + length > SIZE:
        raise InvalidPointerError(f"Address out of range: {addr} len={length}")
    return addr, length

def alloc(size, name):
    """
    Allocate 'size' bytes and bind the base address to custom_var_addrs[name].
    Usage in assembly: alloc 16, pptr  -> creates custom_var_addrs['pptr'] = <addr>
    """
    ptr = memory.malloc(size)
    custom_var_addrs[name] = ptr
    return ptr

def free(name):
    """
    Free a previously allocated symbolic var name (e.g. 'pptr').
    """
    if name not in custom_var_addrs:
        raise InvalidFreeError(f"No such allocated var: {name}")
    ptr = custom_var_addrs.pop(name)
    memory.free(ptr)
    return ptr
def to_int(x):
    """
    Resolve an operand to a Python int:
     - ints return unchanged
     - register names in vars -> their numeric value
     - custom_var_addrs names -> the stored address (int)
     - memory refs [base+offset,len] -> little-endian integer read
     - numeric literals in hex/bin/oct/decimal
     - Ns<string> format -> return integer from UTF-8 string (padded with nulls, little-endian)
       supports escapes: \\c (comma), \\s (space), \\n (newline)
    """
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        s = x.strip()

        # Ns<string> format
        m = re.match(r'^(\d+)s(.+)$', s)
        if m:
            length = int(m.group(1))
            text = m.group(2)

            # handle escapes
            text = (
                text.replace(r"\c", ",")
                    .replace(r"\s", " ")
                    .replace(r"\n", "\n")
            )

            data = text.encode("utf-8")
            if len(data) > length:
                raise ValueError(f"String too long for {length}s format")

            padded = data.ljust(length, b"\x00")
            return int.from_bytes(padded, "little", signed=False)

        if s.startswith("0x"):
            return int(s, 16)
        if s.startswith("0b"):
            return int(s, 2)
        if s.startswith("0o"):
            return int(s, 8)
        if s in vars:
            return vars[s]
        if s in custom_var_addrs:
            return custom_var_addrs[s]
        if s.startswith("[") and s.endswith("]"):
            addr, length = parse_address(s)
            return int.from_bytes(memory.memory[addr:addr+length], "little", signed=False)

        # fallback decimal
        return int(s, 0)

    return int(x)


def mov(dest, src):
    """
    mov(dest, src):
      - dest can be register name -> assign integer value (masked to 64-bit)
      - dest can be a symbolic var name -> assign address/value into custom_var_addrs
      - dest can be memory ref [..] -> writes N-byte little-endian (no masking, arbitrary width)
    """
    val = to_int(src)

    # case 1: register
    if isinstance(dest, str) and dest in vars:
        vars[dest] = val & ((1 << 64) - 1)  # force 64-bit wraparound
        return

    # case 2: symbolic var name
    if isinstance(dest, str) and dest in custom_var_addrs:
        custom_var_addrs[dest] = val  # store full int
        return

    # case 3: memory reference [..]
    if isinstance(dest, str) and dest.startswith("[") and dest.endswith("]"):
        addr, length = parse_address(dest)
        memory.memory[addr:addr+length] = int(val).to_bytes(length, "little", signed=False)
        return

    raise InvalidPointerError(f"Invalid destination: {dest}")

def push(value, stack):
    """
    push(value, stack):
      - if stack is a registered stack var in vars (list), append numeric value
      - if stack is a memory ref [..], write N-byte value at that address (overwrites)
    """
    if isinstance(stack, str) and stack in vars:
        val = to_int(value)
        vars[stack].append(val)
        return

    if isinstance(stack, str) and stack.startswith("[") and stack.endswith("]"):
        addr, length = parse_address(stack)
        val = to_int(value)
        memory.memory[addr:addr+length] = int(val).to_bytes(length, "little", signed=False)
        return

    raise InvalidPointerError(f"Invalid stack: {stack}")


def pop(stack, dest):
    """
    pop(stack, dest):
      - if stack is a registered stack var in vars (list), pop and place into dest
      - if stack is a memory ref [..], read N bytes from that memory as the 'popped' value
      - dest may be register name, memory ref, or custom_var_addrs name
    """
    if isinstance(stack, str) and stack in vars:
        if not vars[stack]:
            raise InvalidPointerError(f"Pop from empty stack: {stack}")
        value = vars[stack].pop()

    elif isinstance(stack, str) and stack.startswith("[") and stack.endswith("]"):
        addr, length = parse_address(stack)
        value = int.from_bytes(memory.memory[addr:addr+length], "little", signed=False)

    else:
        raise InvalidPointerError(f"Invalid stack: {stack}")

    # write to dest
    if isinstance(dest, str) and dest in vars:
        vars[dest] = value & ((1 << 64) - 1)  # mask for registers
        return

    if isinstance(dest, str) and dest in custom_var_addrs:
        custom_var_addrs[dest] = value
        return

    if isinstance(dest, str) and dest.startswith("[") and dest.endswith("]"):
        addr, length = parse_address(dest)
        memory.memory[addr:addr+length] = int(value).to_bytes(length, "little", signed=False)
        return

    raise InvalidPointerError(f"Invalid destination: {dest}")
def add(dest, src1, src2):
    mov(dest, to_int(src1) + to_int(src2))

def sub(dest, src1, src2):
    mov(dest, to_int(src1) - to_int(src2))

def mul(dest, src1, src2):
    mov(dest, to_int(src1) * to_int(src2))

def div(dest, src1, src2):
    b = to_int(src2)
    if b == 0:
        raise ZeroDivisionError("Division by zero in div()")
    mov(dest, to_int(src1) // b)

def mod(dest, src1, src2):
    b = to_int(src2)
    if b == 0:
        raise ZeroDivisionError("Modulo by zero in mod()")
    mov(dest, to_int(src1) % b)
def call_int(x):
    interrupt = to_int(x)

    if interrupt == 0x00:
        # print AX as number
        print(str(vars[AX] & 0xFF), end='', flush=True)

    elif interrupt == 0x01:
        # print AX as char
        print(chr(vars[AX] & 0xFF), end='', flush=True)

    elif interrupt == 0x02:
        # print string from memory[AX:AX+BX]
        addr = vars[AX]
        length = vars[BX]
        data = memory.memory[addr:addr+length]
        text = data.rstrip(b"\x00").decode("utf-8", errors="ignore")
        print(text, end='', flush=True)

    elif interrupt == 0x03:
        # read file from filepath memory
        filepath_addr = vars[AX]
        filepath_len  = vars[BX]
        read_len      = vars[CX]
        out_addr      = vars[DX]

        path_bytes = memory.memory[filepath_addr:filepath_addr+filepath_len].rstrip(b"\x00")
        filepath = path_bytes.decode("utf-8", errors="ignore")

        try:
            with open(filepath, "rb") as f:
                data = f.read(read_len)
            memory.memory[out_addr:out_addr+len(data)] = data
            # optionally zero-pad remaining bytes
            if len(data) < read_len:
                memory.memory[out_addr+len(data):out_addr+read_len] = b"\x00"*(read_len-len(data))
            vars[AX] = len(data)  # return bytes read in AX
        except Exception as e:
            vars[AX] = 0
            print(f"[INT 0x03 ERROR] {e}")

    elif interrupt == 0x04:
        # write file from memory
        filepath_addr = vars[AX]
        filepath_len  = vars[BX]
        data_addr     = vars[CX]
        data_len      = vars[DX]

        path_bytes = memory.memory[filepath_addr:filepath_addr+filepath_len].rstrip(b"\x00")
        filepath = path_bytes.decode("utf-8", errors="ignore")

        try:
            data = memory.memory[data_addr:data_addr+data_len]
            with open(filepath, "wb") as f:
                f.write(data)
            vars[AX] = data_len  # bytes written
        except Exception as e:
            vars[AX] = 0
            print(f"[INT 0x04 ERROR] {e}")

    elif interrupt == 0x05:
        # blocking single char input, read immediately without Enter
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)               # set terminal to raw mode
            ch = sys.stdin.read(1)       # read exactly one char
            vars[AX] = ord(ch) if ch else 0
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    elif interrupt == 0xFF:
        # exit
        print()
        raise SystemExit(0)

import sys, tty, termios
def label(name):
    labels[name] = vars[PC]
def jmp(label_name):
    if label_name not in labels:
        raise InvalidPointerError(f"No such label: {label_name}")
    mov(PC, labels[label_name])
def jcnd(label_name):
    if vars[CMPF] != 0:
        jmp(label_name)
def jncnd(label_name):
    if vars[CMPF] == 0:
        jmp(label_name)
def cmp(mode, val1, val2):
    if mode == 0:  # equal
        vars[CMPF] = 1 if to_int(val1) == to_int(val2) else 0
    elif mode == 1:  # not equal
        vars[CMPF] = 1 if to_int(val1) != to_int(val2) else 0
    elif mode == 2:  # less than
        vars[CMPF] = 1 if to_int(val1) < to_int(val2) else 0
    elif mode == 3:  # greater than
        vars[CMPF] = 1 if to_int(val1) > to_int(val2) else 0
    elif mode == 4:  # less or equal
        vars[CMPF] = 1 if to_int(val1) <= to_int(val2) else 0
    elif mode == 5:  # greater or equal
        vars[CMPF] = 1 if to_int(val1) >= to_int(val2) else 0
    else:
        raise ValueError(f"Invalid cmp mode: {mode}")
def function(name, code):
    functions[name] = vars[PC]
    pc = vars[PC]
    while pc < len(code):
        if "end" not in code[pc]:
            pc += 1
            continue
        break
    mov(PC, pc)
def check_dependencies(file):
    dependencies = set()
    for i in file.split("\n"):
        i = i.strip()
        if i.startswith("using "):
            dependency = i.split(None, 1)[1].strip()
            dependencies.add(dependency)
            dependencies.update(check_dependencies(open(dependency).read()))
    return dependencies
def run(code):
    global memory, vars, labels, custom_var_addrs, functions
    vars = {"ax": 0, "bx": 0, "cx": 0, "dx": 0, "pstack": [], "stack": [], "astack": [], "cstack": [], "cmpf": 0, "pc": 0}
    dependencies = check_dependencies(code)
    for dep in dependencies:
        with open(dep, "r") as f:
            dep_code = f.read()
            code = dep_code + "\n" + code
    code = code.split("\n")
    mov(PC, 0)
    for i, line in enumerate(code):
        if line.startswith("label "):
            _, name = line.split(None, 1)
            labels[name] = i

    while vars[PC] < len(code):
        line = code[vars[PC]].strip()
        if not line or line.startswith(";"):
            mov(PC, vars[PC] + 1)
            continue
        parts = re.findall(r'\[.*?\]|[^,\s]+', line)
        instr = parts[0].lower()
        args = parts[1:]
        if instr == "mov":
            mov(args[0], args[1])
        elif instr == "add":
            add(args[0], args[1], args[2])
        elif instr == "sub":
            sub(args[0], args[1], args[2])
        elif instr == "mul":
            mul(args[0], args[1], args[2])
        elif instr == "div":
            div(args[0], args[1], args[2])
        elif instr == "mod":
            mod(args[0], args[1], args[2])
        elif instr == "push":
            push(args[0], args[1])
        elif instr == "pop":
            pop(args[0], args[1])
        elif instr == "alloc":
            alloc(int(args[0]), args[1])
        elif instr == "free":
            free(args[0])
        elif instr == "realloc":
            ptr = custom_var_addrs.get(args[0])
            if ptr is None:
                raise InvalidPointerError(f"No such allocated var: {args[0]}")
            new_ptr = memory.realloc(ptr, int(args[1]))
            custom_var_addrs[args[0]] = new_ptr
        elif instr == "int":
            call_int(args[0])
        elif instr == "label":
            #label(args[0])
            pass
        elif instr == "using":
            pass
        elif instr == "jmp":
            jmp(args[0])
        elif instr == "jcnd":
            jcnd(args[0])
        elif instr == "jncnd":
            jncnd(args[0])
        elif instr == "cmp":
            cmp(int(args[0]), args[1], args[2])
        elif instr == "func":
            function(args[0], code)
        elif instr == "end":
            push(DX, STACK)
            pop(CSTACK, DX)
            mov(PC, vars[DX])
            pop(STACK, DX)
        elif instr == "call":
            if args[0] not in functions:
                raise InvalidPointerError(f"No such function: {args[0]}")
            vars[CSTACK].append(vars[PC])
            mov(PC, functions[args[0]])
        else:
            raise ValueError(f"Unknown instruction: {instr}")
        mov(PC, vars[PC] + 1)
        #print(vars, custom_var_addrs)
#!/usr/bin/env python3
import re
REG_IDS = {
    "ax": 0,
    "bx": 1,
    "cx": 2,
    "dx": 3,
    "cmpf": 4,
    "pc": 5,
}

STACK_IDS = {
    "pstack": 6,
    "stack": 7,
    "astack": 8,
    "cstack": 9,
}
var_table: dict[str, int] = {}
custom_var_addrs: dict[str, int] = {}

# ---------------------------
# Exceptions
# ---------------------------
class InvalidPointerError(Exception): pass


# ---------------------------
# Token
# ---------------------------
class TokenType:
    LITERAL = "LITERAL"
    REG     = "REG"
    STACK   = "STACK"
    MEM     = "MEM"
    PTR     = "PTR"
    CUSTOM  = "CUSTOM"


class Token:
    def __init__(self, type_, raw_value, parsed_value=None):
        self.type = type_
        self.raw_value = raw_value   # unparsed string form
        self.parsed_value = parsed_value  # resolved int/addr/etc.

    def __repr__(self):
        return f"Token({self.type}, raw={self.raw_value}, parsed={self.parsed_value})"


# ---------------------------
# Instruction
# ---------------------------
class Instruction:
    def __init__(self, opcode, arg1=None, arg2=None, arg3=None):
        self.opcode = opcode
        self.arg1 = arg1  # Token or None
        self.arg2 = arg2
        self.arg3 = arg3

    def __repr__(self):
        return f"Instruction({self.opcode}, {self.arg1}, {self.arg2}, {self.arg3})"


# ---------------------------
# Address parser (MEM)
# ---------------------------
def parse_address(expr: str) -> tuple[int, int]:
    """
    Parse memory reference like [ptr+8,16] or [0x1000-4,4].
    Returns (absolute_address, length).
    """
    if not (isinstance(expr, str) and expr.startswith("[") and expr.endswith("]")):
        raise InvalidPointerError(f"Invalid memory reference: {expr}")

    inside = expr[1:-1].strip()
    # support optional ,len
    if "," in inside:
        base_part, len_part = inside.split(",", 1)
        length = int(len_part.strip())
    else:
        base_part, length = inside, 8  # default = 8 bytes

    m = re.match(r"(.+?)([+-]\d+)?$", base_part.strip())
    if not m:
        raise InvalidPointerError(f"Invalid memory address: {expr}")

    base_str, offset_str = m.groups()
    base_str = base_str.strip()

    # resolve base
    if base_str in vars:
        base = vars[base_str]
    elif base_str in custom_var_addrs:
        base = custom_var_addrs[base_str]
    else:
        try:
            base = to_int(base_str)
        except Exception:
            raise InvalidPointerError(f"Invalid base in memory reference: {base_str}")

    offset = int(offset_str) if offset_str else 0
    addr = base + offset

    if addr < 0 or addr + length > SIZE:
        raise InvalidPointerError(f"Address out of range: {addr} len={length}")
    return addr, length

# ---------------------------
# Parser (ASM -> AST)
# ---------------------------
REGS   = {"ax", "bx", "cx", "dx", "cmpf", "pc"}
STACKS = {"pstack", "stack", "astack", "cstack"}


def guess_token_type(arg: str) -> str:
    """Heuristically determine TokenType from raw argument string."""
    if arg in REGS:
        return TokenType.REG
    if arg in STACKS:
        return TokenType.STACK
    if arg.startswith("[") and arg.endswith("]"):
        return TokenType.MEM
    if re.match(r"^\d+s.+$", arg):  # Ns<string>
        return TokenType.LITERAL
    if arg.startswith("0x") or arg.startswith("0b") or arg.startswith("0o") or arg.isdigit():
        return TokenType.LITERAL
    return TokenType.CUSTOM


def parse_line(line: str) -> Instruction | None:
    """Parse a single ASM line into an Instruction object (or None)."""
    line = line.strip()
    if not line or line.startswith(";"):
        return None

    parts = re.findall(r'\[.*?\]|[^,\s]+', line)
    opcode = parts[0].lower()
    args = parts[1:]

    tokens = []
    for arg in args:
        ttype = guess_token_type(arg)
        parsed = None
        try:
            parsed = to_int(arg)
        except Exception:
            parsed = arg
            if parsed in REG_IDS:
                parsed = REG_IDS[parsed]
            elif parsed in STACK_IDS:
                parsed = STACK_IDS[parsed]
            elif arg.startswith("[") and arg.endswith("]"):
                inside = arg[1:-1].strip()

                if "," in inside:
                    base_offset, length_str = inside.split(",", 1)
                    length_token = Token(TokenType.LITERAL, length_str.strip(), to_int(length_str.strip()))
                else:
                    base_offset, length_token = inside, Token(TokenType.LITERAL, "8", 8)  # default length 8

                m = re.match(r"(.+?)([+-]\d+)?$", base_offset.strip())
                if not m:
                    raise ValueError(f"Invalid memory reference: {arg}")

                base_str, offset_str = m.groups()

                # base can be CUSTOM, REG, or LITERAL
                if base_str in REG_IDS:
                    base_token = Token(TokenType.REG, base_str, REG_IDS[base_str])
                elif base_str in STACK_IDS:
                    base_token = Token(TokenType.STACK, base_str, STACK_IDS[base_str])
                elif re.match(r'^(0x|0b|0o|\d+)', base_str):
                    base_token = Token(TokenType.LITERAL, base_str, to_int(base_str))
                else:
                    base_token = Token(TokenType.CUSTOM, base_str, base_str)

                if offset_str:
                    offset_token = Token(TokenType.LITERAL, offset_str, int(offset_str))
                else:
                    offset_token = Token(TokenType.LITERAL, "0", 0)

                mp = MemoryPointer(base_token, offset_token, length_token)
                tokens.append(Token(TokenType.MEM, arg, mp))
                continue

        tokens.append(Token(ttype, arg, parsed))

    # pad to 3 args
    while len(tokens) < 3:
        tokens.append(None)

    return Instruction(opcode, tokens[0], tokens[1], tokens[2])

class MemoryPointer:
    def __init__(self, base: Token, offset: Token, length: Token):
        self.base = base      # CUSTOM | REG | LITERAL
        self.offset = offset  # LITERAL
        self.length = length  # LITERAL

    def __repr__(self):
        return f"MemoryPointer(base={self.base}, offset={self.offset}, length={self.length})"

def parse_program(code: str) -> tuple[list[Instruction], dict[str, int]]:
    """Parse full ASM program text into a list of Instructions + labels dict."""
    instructions = []
    labels = {}

    lines = code.splitlines()
    for i, line in enumerate(lines):
        instr = parse_line(line)
        if instr is None:
            continue
        instructions.append(instr)

        if instr.opcode == "label" and instr.arg1:
            labels[instr.arg1.raw_value] = len(instructions) - 1

    return instructions, labels

OPCODES = {
    # Data movement
    "mov":   0x01,
    "push":  0x02,
    "pop":   0x03,
    "alloc": 0x04,
    "free":  0x05,
    "realloc": 0x06,

    # Arithmetic
    "add":   0x07,
    "sub":   0x08,
    "mul":   0x09,
    "div":   0x0A,
    "mod":   0x0B,
    "cmp":   0x0C,

    # Control flow
    "jmp":   0x0D,
    "jcnd":  0x0E,
    "jncnd": 0x0F,
    "label": 0x10,  # not emitted, just a marker
    "func":  0x11,
    "call":  0x12,
    "end":   0x13,

    # System
    "int":   0x14,
    "using": 0x15,  # optional directive
}


TYPE_TAGS = {
    TokenType.REG:    0x01,
    TokenType.STACK:  0x02,
    TokenType.LITERAL:0x03,
    TokenType.MEM:    0x04,
    TokenType.CUSTOM: 0x05,
}

def encode_int(value: int, width: int) -> bytes:
    return value.to_bytes(width, "little", signed=True)

def minimal_width(value: int) -> int:
    if -128 <= value <= 127: return 1
    if -32768 <= value <= 32767: return 2
    if -2147483648 <= value <= 2147483647: return 4
    return 8
def encode_token(tok, symtab):
    if tok is None:
        return b""

    ttag = TYPE_TAGS.get(tok.type)
    if ttag is None:
        raise ValueError(f"Unhandled token type {tok.type}")

    if tok.type == TokenType.REG:
        return bytes([ttag, 1, tok.parsed_value])

    elif tok.type == TokenType.STACK:
        # STACK tokens encoded same as REG: type tag, width=1, ID
        return bytes([ttag, 1, tok.parsed_value])

    elif tok.type == TokenType.LITERAL:
        width = minimal_width(tok.parsed_value)
        return bytes([ttag, width]) + encode_int(tok.parsed_value, width)

    elif tok.type == TokenType.CUSTOM:
        # always encode as string for labels and using
        raw_bytes = tok.parsed_value.encode("utf-8")
        return bytes([ttag, len(raw_bytes)]) + raw_bytes

    elif tok.type == TokenType.MEM:
        mp = tok.parsed_value
        bpart = encode_token(mp.base, symtab)
        opart = encode_token(mp.offset, symtab)
        lpart = encode_token(mp.length, symtab)
        return bytes([ttag, len(bpart)+len(opart)+len(lpart)]) + bpart + opart + lpart

    raise ValueError(f"Unhandled token type {tok.type}")

def encode_instruction(instr: Instruction, symtab: dict[str,int]) -> bytes:
    opcode = OPCODES.get(instr.opcode)
    if opcode is None:
        raise ValueError(f"Unknown opcode {instr.opcode}")

    out = bytes([opcode])
    for tok in (instr.arg1, instr.arg2, instr.arg3):
        if tok is not None:
            out += encode_token(tok, symtab)
    return out

def compile_program(instrs: list[Instruction], labels: dict[str,int]) -> bytes:
    """Turn full program into bytecode."""
    # symbol table = labels + custom vars
    symtab = {**labels}  # copy labels
    out = b""
    for instr in instrs:
        # always emit the instruction, including label and using
        out += encode_instruction(instr, symtab)

    return out
# ---------------------------
# Decompiler + round-trip check (replace previous broken code)
# ---------------------------

# reverse maps
REVERSE_REG_IDS = {v: k for k, v in REG_IDS.items()}
REVERSE_STACK_IDS = {v: k for k, v in STACK_IDS.items()}
REVERSE_OPCODES = {v: k for k, v in OPCODES.items()}
REVERSE_TYPES = {v: k for k, v in TYPE_TAGS.items()}

# how many operands each opcode uses
OPERAND_COUNTS = {
    # Data movement
    "mov":    2,
    "push":   2,
    "pop":    2,
    "alloc":  2,
    "free":   1,
    "realloc":2,

    # Arithmetic
    "add":    3,
    "sub":    3,
    "mul":    3,
    "div":    3,
    "mod":    3,
    "cmp":    3,

    # Control flow
    "jmp":    1,
    "jcnd":   1,
    "jncnd":  1,
    "label":  1,
    "func":   1,
    "call":   1,
    "end":    0,

    # System
    "int":    1,
    "using":  1,  # optional directive
}


def read_uint_le(data: bytes, off: int, width: int) -> tuple[int,int]:
    if width == 0:
        return 0, off
    val = int.from_bytes(data[off:off+width], "little", signed=False)
    return val, off + width

def decode_token_bytes_to_ast(data: bytes, off: int) -> tuple[Token,int]:
    """
    Decode one token from bytecode starting at off and return a Token instance
    (using your Token class) and new offset.
    """
    if off + 2 > len(data):
        raise EOFError("Unexpected EOF while decoding token header")
    type_tag = data[off]; off += 1
    width = data[off]; off += 1

    ttype = REVERSE_TYPES.get(type_tag)
    # REG
    if ttype == TokenType.REG:
        val, off = read_uint_le(data, off, width)
        # map id back to name if known
        name = REVERSE_REG_IDS.get(val, f"r{val}")
        return Token(TokenType.REG, name, val), off

    # STACK
    if ttype == TokenType.STACK:
        val, off = read_uint_le(data, off, width)
        name = REVERSE_STACK_IDS.get(val, f"stk{val}")
        return Token(TokenType.STACK, name, val), off

    # LITERAL
    if ttype == TokenType.LITERAL:
        val, off = read_uint_le(data, off, width)
        # keep raw_value as the canonical literal formatter (hex or Ns later)
        return Token(TokenType.LITERAL, str(val), val), off

    # CUSTOM (symbol name stored as raw utf-8 bytes)
    if ttype == TokenType.CUSTOM:
        raw = data[off:off+width]; off += width
        name = raw.decode("utf-8", errors="replace")
        return Token(TokenType.CUSTOM, name, name), off

    # MEM: payload contains nested tokens; parse until we've consumed width bytes
    if ttype == TokenType.MEM:
        end = off + width
        # decode sub-tokens into Token objects
        base_tok, off = decode_token_bytes_to_ast(data, off)
        offset_tok, off = decode_token_bytes_to_ast(data, off)
        length_tok, off = decode_token_bytes_to_ast(data, off)
        if off != end:
            # if there are extra padding bytes, just skip them (tolerant)
            off = end
        mp = MemoryPointer(base_tok, offset_tok, length_tok)
        return Token(TokenType.MEM, f"[{base_tok.raw_value}+{offset_tok.parsed_value},{length_tok.parsed_value}]", mp), off

    # fallback - unknown type tag: consume width bytes as raw
    raw = data[off:off+width]; off += width
    return Token(TokenType.CUSTOM, raw.hex(), raw.hex()), off

def decode_instruction_bytes_to_ast(data: bytes, off: int) -> tuple[Instruction,int]:
    """Decode a single instruction into Instruction (AST) and return new offset."""
    if off >= len(data):
        raise EOFError("EOF when expecting instruction")
    opcode_b = data[off]; off += 1
    opname = REVERSE_OPCODES.get(opcode_b, f"op_{opcode_b:02x}")

    nargs = OPERAND_COUNTS.get(opname, 0)
    toks = []
    for _ in range(nargs):
        if off >= len(data):
            raise EOFError("Unexpected EOF while decoding operands")
        tok, off = decode_token_bytes_to_ast(data, off)
        toks.append(tok)

    # pad to 3
    while len(toks) < 3:
        toks.append(None)

    return Instruction(opname, toks[0], toks[1], toks[2]), off


def decompile_program_to_ast(bytecode: bytes) -> list[Instruction]:
    instrs = []
    off = 0
    while off < len(bytecode):
        ins, off = decode_instruction_bytes_to_ast(bytecode, off)
        instrs.append(ins)
    return instrs
def format_instruction_as_asm(ins: Instruction) -> str:
    parts = [ins.opcode]
    for arg in (ins.arg1, ins.arg2, ins.arg3):
        if arg is None:
            continue
        if arg.type == TokenType.MEM:
            mp: MemoryPointer = arg.parsed_value
            base = mp.base.raw_value      # use raw_value here
            offset_val = mp.offset.parsed_value if mp.offset else 0
            length_val = mp.length.parsed_value if mp.length else 0
            parts.append(f"[{base}+{offset_val},{length_val}]")
        else:
            parts.append(str(arg.raw_value))  # use raw_value, not parsed_value
    return parts[0] + (" " + ", ".join(parts[1:]) if len(parts) > 1 else "")

def format_program_from_ast(instrs: list[Instruction]) -> str:
    return "\n".join(format_instruction_as_asm(i) for i in instrs)

def normalize_program_text(src: str) -> list[str]:
    """Normalize source text: strip, remove empty lines, collapse spaces, lowercase opcodes for fair compare."""
    out = []
    for line in src.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        # collapse multiple spaces, normalize commas spacing
        line = re.sub(r"\s*,\s*", ", ", line)
        line = re.sub(r"\s+", " ", line)
        # lowercase opcode
        if " " in line:
            op, rest = line.split(" ", 1)
            out.append(op.lower() + " " + rest)
        else:
            out.append(line.lower())
    return out

def memorypointer_to_tuple(mp: MemoryPointer):
    return (
        mp.base.parsed_value if hasattr(mp.base, "parsed_value") else mp.base,
        mp.offset.parsed_value if hasattr(mp.offset, "parsed_value") else mp.offset,
        mp.length.parsed_value if hasattr(mp.length, "parsed_value") else mp.length,
    )


def ast_to_tuple(instr: Instruction):
    def tok_to_val(tok):
        if tok is None:
            return None
        if isinstance(tok.parsed_value, MemoryPointer):
            return memorypointer_to_tuple(tok.parsed_value)
        if hasattr(tok, "parsed_value"):
            return tok.parsed_value
        return tok
    return (
        instr.opcode,
        tok_to_val(instr.arg1),
        tok_to_val(instr.arg2),
        tok_to_val(instr.arg3),
    )


import argparse
import sys
from pathlib import Path
import os
def resolve_dependencies(instrs, loaded_modules=None, base_path="."):
    """Recursively inline modules for 'using' instructions."""
    if loaded_modules is None:
        loaded_modules = set()

    final_instrs = []
    for ins in instrs:
        if ins.opcode == "using":
            module_name = ins.arg1.parsed_value
            if module_name in loaded_modules:
                continue  # already loaded
            loaded_modules.add(module_name)

            # Convert module name to file path
            mod_path = Path(base_path) / Path(module_name.replace(".", "/") + ".kasm")
            if not mod_path.exists():
                raise FileNotFoundError(f"Module '{module_name}' not found at {mod_path}")
            
            with open(mod_path, "r", encoding="utf-8") as f:
                mod_src = f.read()

            mod_instrs, _ = parse_program(mod_src)
            # Recurse to resolve nested modules
            mod_instrs = resolve_dependencies(mod_instrs, loaded_modules, base_path)
            final_instrs.extend(mod_instrs)
        else:
            final_instrs.append(ins)

    return final_instrs

def main():
    parser = argparse.ArgumentParser(description="KASM VM Runner")
    parser.add_argument("input", help="Input bytecode (.kbc) file")
    parser.add_argument("-v", "--verify", action="store_true", help="Decompile and verify AST")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file '{args.input}' not found")
        return

    # Read bytecode file
    bytecode = input_path.read_bytes()

    # Decompile to AST
    decoded_instrs = decompile_program_to_ast(bytecode)

    if args.verify:
        # Print decompiled ASM
        print("\n--- Decompiled ASM ---")
        for ins in decoded_instrs:
            print(format_instruction_as_asm(ins))

    # Convert AST back to code string for `run()`
    code_lines = [format_instruction_as_asm(ins) for ins in decoded_instrs]
    code_str = "\n".join(code_lines)

    # Run in VM
    run(code_str)

if __name__ == "__main__":
    main()