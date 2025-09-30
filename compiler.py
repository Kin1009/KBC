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
# to_int (operand resolver)
# ---------------------------
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
        if s.startswith("[") and s.endswith("]"):
            addr, length = parse_address(s)
            return int.from_bytes(memory.memory[addr:addr+length], "little", signed=False)

        # fallback decimal
        return int(s, 0)

    return int(x)


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
    parser = argparse.ArgumentParser(description="KASM AST compiler/decompiler")
    parser.add_argument("input", help="Input .kasm file")
    parser.add_argument("-o", "--output", help="Output bytecode file")
    parser.add_argument("-v", "--verify", action="store_true", help="Verify round-trip AST")
    args = parser.parse_args()

    # Read input file
    with open(args.input, "r", encoding="utf-8") as f:
        program_src = f.read()

    # Parse program
    instrs, labels = parse_program(program_src)
    instrs = resolve_dependencies(instrs, base_path=os.path.dirname(args.input))

    # Compile to bytecode
    bytecode = compile_program(instrs, labels)
    if args.output:
        with open(args.output, "wb") as f:
            f.write(bytecode)
    else:
        print("Bytecode:", bytecode.hex(" "))

    #print("Labels:", labels)

    # Verification
    if args.verify:
        decoded_instrs = decompile_program_to_ast(bytecode)
        orig_ast = [ast_to_tuple(i) for i in instrs]
        dec_ast  = [ast_to_tuple(i) for i in decoded_instrs]

        print("\n--- AST Round-trip ---")
        if orig_ast == dec_ast:
            print("SUCCESS: Decompiled AST matches original AST.")
        else:
            print("MISMATCH:")
            from pprint import pprint
            print("Original AST:")
            pprint(orig_ast)
            print("Decompiled AST:")
            pprint(dec_ast)

    # Example decompiled ASM output
    #print("\n--- Decompiled ASM ---")
    #decoded_instrs = decompile_program_to_ast(bytecode)
    #for ins in decoded_instrs:
    #    print(format_instruction_as_asm(ins))

if __name__ == "__main__":
    main() # type: ignore