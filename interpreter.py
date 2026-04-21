#!/usr/bin/env python3
"""
Interpreter for the GraalVM build scripting language.
Walks the AST produced by parser.py and executes each node.

Usage
-----
    python interpreter.py source.script
    python interpreter.py --dry-run source.script   # print commands, don't run them
    python interpreter.py --trace source.script      # print each statement before running
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from typing import Optional

# Import AST nodes and parse() from the sibling parser module.
# Adjust the path if parser.py lives elsewhere.
sys.path.insert(0, os.path.dirname(__file__))
from parser import (
    CallExpr, CatchBlock, ContinueStmt, DottedName, FuncDef,
    GlobalDecl, GlobalDefsBlock, Identifier, NumberLit, Program,
    ReturnStmt, StringLit, Param,
    parse,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal control-flow signals  (not user-visible exceptions)
# ═══════════════════════════════════════════════════════════════════════════════

class _ReturnSignal(Exception):
    """Raised by return(val) to unwind the call stack."""
    def __init__(self, value: str) -> None:
        self.value = value

class _ExitSignal(Exception):
    """Raised by exitScript(code) to terminate the process."""
    def __init__(self, code: int) -> None:
        self.code = code


# ═══════════════════════════════════════════════════════════════════════════════
#  Interpreter
# ═══════════════════════════════════════════════════════════════════════════════

class Interpreter:
    """
    Execution model
    ───────────────
    • GlobalDefs entries are loaded into self.globals as the initial variable
      table.  'global NAME = expr' statements add to the same table at runtime.

    • Every function call sets self.last_ok (True = success, False = failure).
      catch(Exception.PrevCmd) { ... } else { ... } branches on that flag.

    • _interp functions shadow built-ins of the same name.  Their 'arg'
      parameters are bound positionally when called.

    • String interpolation ($VAR, ${VAR}) is performed on every string value
      just before it is used, using the current state of self.globals.

    • mapvar("$X", "BASH-VAR")      -> export X's value to the bash env as X
      mapvar("$X", "BASH-VAR:ALIAS")-> export X's value to the bash env as ALIAS
      Both also add the alias to self.globals so later $ALIAS interpolation works.
    """

    def __init__(
        self,
        program: Program,
        *,
        dry_run: bool = False,
        trace:   bool = False,
    ) -> None:
        self.program  = program
        self.dry_run  = dry_run
        self.trace    = trace

        self.globals:  dict[str, str]    = {}   # script-level variables
        self.functions: dict[str, FuncDef] = {}  # name -> FuncDef
        self.bash_env: dict[str, str]    = {}   # extra vars injected into bash

        self.last_ok:  bool = True   # result of the most recent command

    def _hoist_globals(self) -> None:
        """Pre-evaluate every GlobalDecl in every function body.

        This ensures variables like MxTool are populated even when the
        function that declares them is skipped by the build-progress cache.
        Functions are processed in source order so later declarations can
        reference earlier ones (e.g. MxTool uses BASEDIR).
        """
        for fn in self.program.functions:
            for stmt in fn.body:
                if isinstance(stmt, GlobalDecl):
                    self.globals[stmt.name] = self._eval(stmt.value, {})

    # ── bootstrap ────────────────────────────────────────────────────────────

    def run(self) -> None:
        # 1. Seed built-in variables
        self.globals["BASEDIR"] = os.getcwd()

        # 2. Load GlobalDefs into the variable table
        if self.program.global_defs:
            self.globals.update(self.program.global_defs.entries)

        # 2. Register all function definitions (including _interp overrides)
        for fn in self.program.functions:
            self.functions[fn.name] = fn

        # 3. Hoist GlobalDecl statements from every function body so that
        #    variables like MxTool are set even when a target is skipped by
        #    the build-progress cache.  GlobalDecls are pure assignments with
        #    no side-effects, so running them up-front is always safe.
        self._hoist_globals()

        # 4. Find and call the entrypoint (ep modifier)
        ep = next(
            (fn for fn in self.program.functions if "ep" in fn.modifiers),
            None,
        )
        if ep is None:
            _die("No entrypoint function (marked with 'ep') found.")

        try:
            self._call_func(ep, [])
        except _ExitSignal as e:
            sys.exit(e.code)

    # ── string interpolation ─────────────────────────────────────────────────

    def _interp(self, s: str, scope: dict | None = None) -> str:
        """Expand $VAR and ${VAR}. Local scope takes priority over globals."""
        env = {**self.globals, **(scope or {})}
        s = re.sub(
            r'\$\{([^}]+)\}',
            lambda m: env.get(m.group(1), m.group(0)),
            s,
        )
        s = re.sub(
            r'\$([A-Za-z_][A-Za-z0-9_]*)',
            lambda m: env.get(m.group(1), m.group(0)),
            s,
        )
        return s

    # ── expression evaluator ─────────────────────────────────────────────────

    def _eval(self, expr, scope: dict) -> str:
        match expr:
            case StringLit(value=v):
                return self._interp(v, scope)
            case NumberLit(value=v):
                return str(v)
            case Identifier(name=n):
                return scope.get(n) or self.globals.get(n, n)
            case CallExpr(name=name, args=args):
                evaled = [self._eval(a, scope) for a in args]
                return self._dispatch(name, evaled, scope)
            case _:
                return str(expr)

    # ── function dispatch ─────────────────────────────────────────────────────

    def _dispatch(self, name: str, args: list[str], scope: dict) -> str:
        """
        Call priority:
          1. _interp function with this name  (user-supplied override)
          2. Built-in implementation
          3. Plain (non-_interp) user-defined function
        """
        fn = self.functions.get(name)
        if fn and "_interp" in fn.modifiers:
            return self._call_func(fn, args)

        return self._builtin(name, args, scope)

    # ── built-in commands ─────────────────────────────────────────────────────

    def _builtin(self, name: str, args: list[str], scope: dict) -> str:  # noqa: C901
        def arg(i: int, default: str = "") -> str:
            return args[i] if i < len(args) else default

        match name:

            # ── I/O ──────────────────────────────────────────────────────────

            case "print":
                msg = arg(0)
                print(msg)
                self.last_ok = True
                return ""

            case "output_str":
                # echo content >> file
                path, content = arg(0), arg(1)
                self._trace(f"output_str({path!r}, {content!r})")
                if not self.dry_run:
                    try:
                        with open(path, "a") as f:
                            f.write(content + "\n")
                        self.last_ok = True
                    except OSError as e:
                        _warn(f"output_str: {e}")
                        self.last_ok = False
                return ""

            case "getcontent":
                # grep: fail (last_ok=False) if content NOT found in file
                path, needle = arg(0), arg(1)
                self._trace(f"getcontent({path!r}, {needle!r})")
                if self.dry_run:
                    self.last_ok = False   # assume not done yet in dry-run
                    return ""
                try:
                    with open(path) as f:
                        self.last_ok = needle in f.read()
                except OSError:
                    self.last_ok = False
                return ""

            # ── file system ──────────────────────────────────────────────────

            case "mkFile":
                directory, filename = arg(0), arg(1)
                path = os.path.join(directory, filename) if directory != "." else filename
                self._trace(f"touch {path}")
                if not self.dry_run:
                    try:
                        open(path, "a").close()
                        self.last_ok = True
                    except OSError as e:
                        _warn(f"mkFile: {e}")
                        self.last_ok = False
                return ""

            case "mkdir" | "mkFolderStructure":
                path = arg(0)
                self._trace(f"mkdir -p {path}")
                if not self.dry_run:
                    try:
                        os.makedirs(path, exist_ok=True)
                        self.last_ok = True
                    except OSError as e:
                        _warn(f"mkdir: {e}")
                        self.last_ok = False
                return ""

            case "ifex":
                # find dir -name file; succeeds if anything matches
                directory, filename = arg(0), arg(1)
                self._trace(f"find {directory} -name {filename!r}")
                if self.dry_run:
                    self.last_ok = False
                    return ""
                matches = glob.glob(os.path.join(directory, filename))
                self.last_ok = bool(matches)
                return ""

            case "chdir":
                path = arg(0)
                self._trace(f"cd {path}")
                if not self.dry_run:
                    try:
                        os.chdir(path)
                        self.last_ok = True
                    except OSError as e:
                        _warn(f"chdir: {e}")
                        self.last_ok = False
                return ""

            # ── shell execution ───────────────────────────────────────────────

            case "bash":
                cmd = arg(0)
                self._trace(f"+ {cmd}")
                if not self.dry_run:
                    # Merge globals into the subprocess environment so shell
                    # variables like $MxTool resolve even without mapvar.
                    safe_globals = {
                        k: v for k, v in self.globals.items()
                        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', k)
                    }
                    env = {**os.environ, **safe_globals, **self.bash_env}
                    result = subprocess.run(cmd, shell=True, env=env)
                    self.last_ok = result.returncode == 0
                return ""

            case "exec":
                cmd = arg(0)
                self._trace(f"exec: {cmd}")
                if self.dry_run:
                    self.last_ok = True
                    return ""
                env = {**os.environ, **self.bash_env}
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, env=env
                )
                self.last_ok = result.returncode == 0
                return result.stdout

            case "wget":
                url = arg(0)
                self._trace(f"+ wget --show-progress {url}")
                if not self.dry_run:
                    # Don't use -q so that 404/network errors are visible.
                    result = subprocess.run(
                        ["wget", "--show-progress", url],
                        env={**os.environ, **self.bash_env},
                    )
                    self.last_ok = result.returncode == 0
                    if not self.last_ok:
                        _warn(f"wget failed for {url}")
                return ""

            case "clone":
                flags, repo = arg(0), arg(1)
                cmd = f"git clone {flags} {repo}"
                self._trace(f"+ {cmd}")
                if not self.dry_run:
                    result = subprocess.run(
                        cmd, shell=True,
                        env={**os.environ, **self.bash_env},
                    )
                    self.last_ok = result.returncode == 0
                return ""

            # ── variable mapping ─────────────────────────────────────────────

            case "mapvar":
                # mapvar("$X", "BASH-VAR")        -> no-op: $X is already in
                #                                    globals, bash strings expand it
                # mapvar("$X", "BASH-VAR:ALIAS")  -> export value of $X into the
                #                                    bash environment as ALIAS, and
                #                                    add ALIAS to globals so $ALIAS
                #                                    interpolation works too
                value = arg(0)   # already interpolated by _eval
                spec  = arg(1)
                if ":" in spec:
                    alias = spec.split(":", 1)[1]
                    self.bash_env[alias] = value
                    self.globals[alias]  = value
                    self._trace(f"export {alias}={value!r}")
                # else: value already reachable via $OriginalVar interpolation
                self.last_ok = True
                return ""

            # ── control ───────────────────────────────────────────────────────

            case "exitScript":
                code = int(arg(0) or "0")
                raise _ExitSignal(code)

            case "return":
                # return() used as a standalone call (outside of ReturnStmt)
                raise _ReturnSignal(arg(0))

            # ── unknown: try plain user-defined function ──────────────────────

            case _:
                fn = self.functions.get(name)
                if fn:
                    return self._call_func(fn, args)
                _warn(f"Unknown function '{name}' — skipping.")
                self.last_ok = False
                return ""

    # ── function call ─────────────────────────────────────────────────────────

    def _call_func(self, fn: FuncDef, args: list[str]) -> str:
        scope = {p.name: (args[i] if i < len(args) else "") for i, p in enumerate(fn.params)}
        try:
            for stmt in fn.body:
                self._exec(stmt, scope)
        except _ReturnSignal as r:
            return r.value
        return ""

    # ── statement executor ────────────────────────────────────────────────────

    def _exec(self, stmt, scope: dict) -> None:
        if self.trace:
            self._trace_stmt(stmt)

        match stmt:

            case ContinueStmt():
                pass   # nothing to do; execution just continues

            case GlobalDecl(name=name, value=val):
                self.globals[name] = self._eval(val, scope)

            case ReturnStmt(value=val):
                raise _ReturnSignal(self._eval(val, scope))

            case CallExpr(name=name, args=args):
                evaled = [self._eval(a, scope) for a in args]
                self._dispatch(name, evaled, scope)

            case CatchBlock(exception=exc, body=body, else_body=else_body):
                # catch(Exception.PrevCmd) branches on the last command's result
                if not self.last_ok:
                    self.last_ok = True   # reset before running handler
                    for s in body:
                        self._exec(s, scope)
                else:
                    for s in else_body:
                        self._exec(s, scope)

            case _:
                _warn(f"Unhandled statement type: {type(stmt).__name__}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _trace(self, msg: str) -> None:
        if self.trace:
            print(f"\033[2m[trace] {msg}\033[0m", file=sys.stderr)

    def _trace_stmt(self, stmt) -> None:
        match stmt:
            case CallExpr(name=n):
                self._trace(f"call  {n}(...)")
            case CatchBlock():
                self._trace(f"catch (last_ok={self.last_ok})")
            case GlobalDecl(name=n):
                self._trace(f"global {n} = ...")
            case ReturnStmt():
                self._trace("return ...")
            case ContinueStmt():
                self._trace("continue")


# ═══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _warn(msg: str) -> None:
    print(f"\033[33m[warn]\033[0m {msg}", file=sys.stderr)

def _die(msg: str) -> None:
    print(f"\033[31m[error]\033[0m {msg}", file=sys.stderr)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    raw_args = sys.argv[1:]
    dry_run  = "--dry-run" in raw_args
    trace    = "--trace"   in raw_args
    files    = [a for a in raw_args if not a.startswith("--")]

    if not files:
        print("Usage: python interpreter.py [--dry-run] [--trace] <source.script>")
        sys.exit(1)

    src  = open(files[0]).read()
    tree = parse(src)
    Interpreter(tree, dry_run=dry_run, trace=trace).run()


if __name__ == "__main__":
    main()
