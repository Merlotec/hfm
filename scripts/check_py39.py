"""Flag Python-3.10+ constructs that break on Dawn (which runs Python 3.9).

Run BEFORE pushing anything that will execute on the cluster:

    python scripts/check_py39.py                # this repo
    python scripts/check_py39.py ../fvm_model ../fvm_solver

Exits non-zero if anything is found, so it can gate a commit.

Why a dedicated tool: `python -m py_compile` is NOT enough.  Some of these are
plain SyntaxErrors and would be caught, but the nastiest ones parse perfectly on
3.9 and only blow up when the line is EXECUTED:

  * `def f(x: int | None)`  — PEP 604 unions are evaluated at def time, so the
    module imports fine locally on 3.11+ and dies at import on 3.9 with
        TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
  * `zip(a, b, strict=True)` — a TypeError only when that line runs, which may
    be hours into a job.

Each finding prints the 3.9-safe replacement.
"""

import ast
import sys
from pathlib import Path

# Names/attributes added after 3.9.
LATER_NAMES = {
    'tomllib': '3.11', 'ExceptionGroup': '3.11', 'BaseExceptionGroup': '3.11',
    'StrEnum': '3.11', 'Self': '3.11', 'assert_type': '3.11',
    'pairwise': '3.10', 'TypeAlias': '3.10', 'ParamSpec': '3.10',
    'TypeGuard': '3.10', 'anext': '3.10', 'aiter': '3.10',
}
FIXES = {
    'pep604': 'use Optional[X] / Union[X, Y] (or add `from __future__ import '
              'annotations` if it is ONLY in annotations)',
    'isinstance-union': 'use isinstance(x, (A, B))',
    'zip-strict': 'drop strict= and assert len(a) == len(b) beforehand',
    'match': 'rewrite as if/elif',
}


def _is_union(node) -> bool:
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)


def _annotations(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if n.returns:
                yield n.returns
            a = n.args
            for arg in list(a.args) + list(a.kwonlyargs) + list(a.posonlyargs):
                if arg.annotation:
                    yield arg.annotation
            for extra in (a.vararg, a.kwarg):
                if extra is not None and extra.annotation:
                    yield extra.annotation
        elif isinstance(n, ast.AnnAssign) and n.annotation:
            yield n.annotation


def check_file(path: Path):
    """Returns [(lineno, kind, detail)] of 3.10+ constructs."""
    out = []
    try:
        src = path.read_text()
    except Exception:
        return out
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [(e.lineno or 0, 'syntax', f'{e.msg} (does not even parse)')]

    # `from __future__ import annotations` defers ANNOTATION evaluation, so PEP
    # 604 is safe there — but NOT anywhere else (aliases, isinstance, casts).
    lazy = any(isinstance(n, ast.ImportFrom) and n.module == '__future__'
               and any(a.name == 'annotations' for a in n.names)
               for n in ast.walk(tree))
    ann_nodes = {id(a) for a in _annotations(tree)}
    in_ann = set()
    for a in _annotations(tree):
        for sub in ast.walk(a):
            in_ann.add(id(sub))

    for n in ast.walk(tree):
        if _is_union(n):
            typish = (any(isinstance(o, ast.Constant) and o.value is None
                          for o in (n.left, n.right))
                      or id(n) in in_ann or id(n) in ann_nodes)
            if typish and not (lazy and id(n) in in_ann):
                where = 'annotation' if id(n) in in_ann else 'runtime position'
                out.append((n.lineno, 'pep604', f'`X | Y` type union in {where}'))
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, 'attr', '')
            if name in ('isinstance', 'issubclass') and len(n.args) > 1:
                if any(_is_union(x) for x in ast.walk(n.args[1])):
                    out.append((n.lineno, 'isinstance-union', f'{name}(x, A | B)'))
            if name == 'zip' and any(k.arg == 'strict' for k in n.keywords):
                out.append((n.lineno, 'zip-strict', 'zip(..., strict=...)'))
        if n.__class__.__name__ == 'Match':
            out.append((n.lineno, 'match', 'match statement'))
        if isinstance(n, ast.Name) and n.id in LATER_NAMES:
            out.append((n.lineno, f'name:{n.id}',
                        f'{n.id} requires Python {LATER_NAMES[n.id]}'))
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mod = getattr(n, 'module', None) or ''
            for a in n.names:
                v = LATER_NAMES.get(a.name) or LATER_NAMES.get(mod)
                if v:
                    out.append((n.lineno, f'import:{a.name}',
                                f'{a.name} requires Python {v}'))
    return sorted(set(out))


def main():
    roots = [Path(a) for a in sys.argv[1:]] or [Path(__file__).resolve().parents[1]]
    total = 0
    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob('*.py'))
        for f in files:
            s = str(f)
            if '__pycache__' in s or 'worktrees' in s or '/.git/' in s:
                continue
            for lineno, kind, detail in check_file(f):
                total += 1
                fix = FIXES.get(kind, '')
                print(f'{f}:{lineno}: [{kind}] {detail}'
                      + (f'\n    -> {fix}' if fix else ''))
    if total:
        print(f'\n{total} construct(s) that will fail on Python 3.9.')
        sys.exit(1)
    print(f'OK — nothing 3.10+ found in {", ".join(str(r) for r in roots)}')


if __name__ == '__main__':
    main()
