#!/usr/bin/env python3
"""
Parser for the Craftfile scripting language.

Grammar summary
---------------
program       → global_defs? func_def*
global_defs   → 'GlobalDefs' '{' (STRING '=' STRING ';')* '}'
func_def      → modifier* IDENT '(' param_list? ')' '{' stmt* '}'
modifier      → 'ep' | '_interp'
param_list    → param (',' param)*
param         → 'arg' IDENT
stmt          → catch_stmt | global_decl | continue_stmt | return_stmt | call_stmt
catch_stmt    → 'catch' '(' dotted_name ')' '{' stmt* '}' ('else' '{' stmt* '}')? ';'
global_decl   → 'global' IDENT '=' expr ';'
continue_stmt → 'continue' ';'
return_stmt   → 'return' '(' expr ')' ';'
call_stmt     → call_expr ';'
call_expr     → IDENT '(' (expr (',' expr)*)? ')'
expr          → STRING | NUMBER | IDENT | call_expr
dotted_name   → IDENT ('.' IDENT)*

Usage
-----
    python parser.py source.script          # pretty-print AST
    python parser.py --tokens source.script # dump token stream
    python parser.py --json source.script   # dump AST as JSON
"""
