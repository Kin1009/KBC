#!/usr/bin/env python3
from lark import Lark, Transformer, v_args

# --- Step 1: Define a C-like grammar ---
c_grammar = r"""
    start: (function | declaration | statement)*

    // Functions
    function: type NAME "(" [params] ")" block
    params: param ("," param)*
    param: type NAME

    // Declarations
    declaration: type NAME ";"

    // Statements
    statement: assignment ";"
              | return_stmt ";"
              | block
              | if_stmt
              | while_stmt
              | ";"

    block: "{" statement* "}"
    assignment: NAME "=" expr
    return_stmt: "return" expr
    if_stmt: "if" "(" expr ")" statement ["else" statement]
    while_stmt: "while" "(" expr ")" statement

    // Expressions
    ?expr: expr "+" term   -> add
         | expr "-" term   -> sub
         | term

    ?term: term "*" factor -> mul
         | term "/" factor -> div
         | factor

    ?factor: NUMBER        -> number
           | NAME          -> var
           | "(" expr ")"

    // Types
    type: "int" | "float" | "void"

    // Tokens
    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""

# --- Step 2: AST Transformer ---
class ASTTransformer(Transformer):
    def start(self, items): return ("program", items)
    def function(self, items): return ("function", items[0], items[1], items[2] if len(items) > 2 else [])
    def param(self, items): return ("param", items[0], items[1])
    def declaration(self, items): return ("decl", items[0], items[1])
    def assignment(self, items): return ("assign", items[0], items[1])
    def return_stmt(self, items): return ("return", items[0])
    def if_stmt(self, items): 
        return ("if", items[0], items[1], items[2] if len(items) > 2 else None)
    def while_stmt(self, items): return ("while", items[0], items[1])
    def block(self, items): return ("block", items)
    def add(self, items): return ("+", items[0], items[1])
    def sub(self, items): return ("-", items[0], items[1])
    def mul(self, items): return ("*", items[0], items[1])
    def div(self, items): return ("/", items[0], items[1])
    def number(self, items): return ("number", str(items[0]))
    def var(self, items): return ("var", str(items[0]))
    def type(self, items): return str(items[0])
    def NAME(self, token): return str(token)

# --- Step 3: Compile parser ---
parser = Lark(c_grammar, parser="lalr", transformer=ASTTransformer())

# --- Step 4: Example usage ---
code = """
int add(int a, int b) {
    int c;
    c = a + b;
    return c;
}
"""

ast = parser.parse(code)
print(ast)
