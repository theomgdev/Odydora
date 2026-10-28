"""Check notebooks/odydora_colab.ipynb the way a notebook reader sees it.

Every assertion here reconstructs a cell with "".join(source), which is what
nbformat specifies and what Jupyter and Colab actually do. Rejoining with
"\\n".join() instead silently supplies newlines the file does not contain, and
that is exactly how a notebook whose every markdown cell rendered as one
run-together line passed a previous check.

Run: python tools/check_notebook.py
"""

import ast
import json
import pathlib
import sys

NB = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "odydora_colab.ipynb"

failures = []


def fail(msg):
    failures.append(msg)


nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]

# 1. Every source line but the last of a cell must end in "\n". This is the
#    whole bug: without it "".join() concatenates the lines.
for i, cell in enumerate(cells):
    src = cell["source"]
    for j, line in enumerate(src[:-1]):
        if not line.endswith("\n"):
            fail(f"cell {i} line {j} has no trailing newline: {line[:60]!r}")
    if src and src[-1].endswith("\n"):
        fail(f"cell {i} last line ends in a newline, which adds a blank line")

# 2. Reconstructed markdown must keep its structure: a heading at the start of
#    a line, a table whose rows are separate lines.
for i, cell in enumerate(cells):
    if cell["cell_type"] != "markdown":
        continue
    text = "".join(cell["source"])
    for marker in ("#", "###", "|---", "- "):
        if marker in text and f"\n{marker}" not in text and not text.startswith(marker):
            fail(f"cell {i}: {marker!r} appears only mid-line, so it will not render")
    # Colab's renderer honours single-tilde strikethrough, so two "~" in one
    # block pair up: "~30 min ... ~2-3 h" struck out everything between them
    # once the lines had collapsed. Guard the class, not just the "~~" form.
    if text.count("~~") % 2:
        fail(f"cell {i}: unbalanced '~~' will render as strikethrough")
    for b_no, block in enumerate(text.split("\n\n")):
        if block.count("~") > 1:
            fail(
                f"cell {i} block {b_no}: {block.count('~')} single '~' in one block "
                f"will pair into strikethrough in Colab -- use 'about' or escape them"
            )

# 3. Code cells must parse, and their non-shell lines must parse together.
n_code = 0
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    n_code += 1
    src = "".join(cell["source"])
    body = "".join(ln for ln in cell["source"] if not ln.lstrip().startswith("!"))
    try:
        ast.parse(body)
    except SyntaxError as e:
        fail(f"code cell {i} does not parse: {e}")
    if "\\\n" in src:
        fail(f"code cell {i} has a backslash continuation, which Colab mangles")

# 4. Imports used in the notebook must exist in the package.
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    body = "".join(ln for ln in cell["source"] if not ln.lstrip().startswith("!"))
    try:
        tree = ast.parse(body)
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("odydora"):
            mod = pathlib.Path(*node.module.split(".")).with_suffix(".py")
            path = NB.resolve().parents[1] / mod
            if not path.exists():
                fail(f"cell {i} imports {node.module}, but {path} does not exist")
                continue
            names = {
                n.name for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(n, ast.FunctionDef | ast.ClassDef)
            }
            for alias in node.names:
                if alias.name not in names:
                    fail(f"cell {i}: {node.module} has no {alias.name}")

# 5. No placeholders left for the reader to resolve.
WHOLE = "".join("".join(c["source"]) for c in cells)
for bad in ("TODO", "FIXME", "replace with your", "once pushed", "by hand"):
    if bad.lower() in WHOLE.lower():
        fail(f"placeholder text {bad!r} left in the notebook")

if failures:
    print(f"FAIL ({len(failures)})")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print(
    f"OK  {len(cells)} cells ({n_code} code), newlines intact, "
    f"markdown structural, imports resolve"
)
