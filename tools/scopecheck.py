"""Scope-aware free-variable check.

Module scope is taken from ``tree.body`` ONLY -- the earlier version walked the
whole tree, so a name imported inside a function looked module-level and a
helper referencing it seemed fine. That is exactly how
``_predict_batch_gpu`` slipped through.
"""
import ast, builtins, sys


def module_scope(tree):
    names = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in tgts:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Name):
                        names.add(sub.id)
        elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            for sub in ast.walk(n):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        names.add(a.asname or a.name.split(".")[0])
                elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    names.add(sub.id)
                elif isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    names.add(sub.name)
    return names


def bound_in(fn):
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n is not fn:
                out.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.comprehension):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
    return out


def check(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    mod = module_scope(tree)
    problems = []

    def visit(fn, enclosing):
        local = bound_in(fn)
        visible = mod | enclosing | local
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in visible:
                problems.append((fn.name, n.id, n.lineno))
        for n in ast.walk(fn):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn:
                pass  # already covered: walk() sees nested bodies within `fn`
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit(n, set())
        elif isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit(m, set())
    return sorted(set(problems))


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        pb = check(p)
        print("%-52s %s" % (p.split("/")[-1], "clean" if not pb else ""))
        for name, ident, line in pb:
            print("    UNDEFINED  %s() reads %r at line %d" % (name, ident, line))
        bad += len(pb)
    sys.exit(1 if bad else 0)
