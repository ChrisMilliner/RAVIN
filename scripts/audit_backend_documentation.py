"""
Audit RAVIN production Python source documentation and formatting.

This development utility performs a static AST-based inspection of
production Python modules. It does not import RAVIN modules and does
not modify repository files.

The audit reports:

- module docstring coverage
- public class docstring coverage
- public function docstring coverage
- public method docstring coverage
- parameter annotation coverage
- return annotation coverage
- RAVIN source-spacing compliance

RAVIN uses a compact project-local Python formatting convention:

- imports remain together without blank lines between import groups
- one blank line follows the import block
- one blank line separates top-level definitions
- one blank line separates class methods

The report establishes a repeatable baseline before bulk
documentation improvements are applied.
"""

from __future__ import annotations
import ast
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    REPOSITORY_ROOT / "backend",
    REPOSITORY_ROOT / "scripts",
)
EXCLUDED_FILES = {
    Path(__file__).resolve(),
}
EXCLUDED_SCRIPT_PREFIXES = (
    "apply_backend_",
)

@dataclass(frozen=True)
class DefinitionAudit:
    """Documentation and typing status for one public definition."""

    file_path: Path
    qualified_name: str
    kind: str
    has_docstring: bool
    parameters_annotated: bool | None
    return_annotated: bool | None

@dataclass(frozen=True)
class ModuleAudit:
    """Documentation status for one Python module."""

    file_path: Path
    has_docstring: bool
    definitions: tuple[
        DefinitionAudit,
        ...,
    ]

@dataclass(frozen=True)
class FormattingIssue:
    """One source-spacing issue detected by the RAVIN audit."""

    file_path: Path
    line_number: int
    message: str

def _is_public(
    name: str,
) -> bool:
    return not name.startswith("_")

def _iter_python_files() -> tuple[
    Path,
    ...,
]:
    files: list[Path] = []

    for root in SOURCE_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*.py"):
            resolved = path.resolve()

            if resolved in EXCLUDED_FILES:
                continue

            if "__pycache__" in path.parts:
                continue

            if (
                path.parent == REPOSITORY_ROOT / "scripts"
                and path.name.startswith(
                    EXCLUDED_SCRIPT_PREFIXES
                )
            ):
                continue

            files.append(
                resolved
            )

    return tuple(
        sorted(files)
    )

def _parameters_are_annotated(
    node: (
        ast.FunctionDef
        | ast.AsyncFunctionDef
    ),
) -> bool:
    arguments = node.args

    positional = (
        *arguments.posonlyargs,
        *arguments.args,
    )

    checked_arguments = [
        argument
        for argument in positional
        if argument.arg not in {
            "self",
            "cls",
        }
    ]

    checked_arguments.extend(
        arguments.kwonlyargs
    )

    if arguments.vararg is not None:
        checked_arguments.append(
            arguments.vararg
        )

    if arguments.kwarg is not None:
        checked_arguments.append(
            arguments.kwarg
        )

    return all(
        argument.annotation is not None
        for argument in checked_arguments
    )

def _audit_function(
    *,
    file_path: Path,
    node: (
        ast.FunctionDef
        | ast.AsyncFunctionDef
    ),
    qualified_name: str,
    kind: str,
) -> DefinitionAudit:
    return DefinitionAudit(
        file_path=file_path,
        qualified_name=qualified_name,
        kind=kind,
        has_docstring=(
            ast.get_docstring(node)
            is not None
        ),
        parameters_annotated=(
            _parameters_are_annotated(
                node
            )
        ),
        return_annotated=(
            node.returns is not None
        ),
    )

def _audit_class(
    *,
    file_path: Path,
    node: ast.ClassDef,
) -> tuple[
    DefinitionAudit,
    ...,
]:
    audits: list[
        DefinitionAudit
    ] = [
        DefinitionAudit(
            file_path=file_path,
            qualified_name=node.name,
            kind="class",
            has_docstring=(
                ast.get_docstring(node)
                is not None
            ),
            parameters_annotated=None,
            return_annotated=None,
        )
    ]

    for child in node.body:
        if not isinstance(
            child,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        if not _is_public(
            child.name
        ):
            continue

        audits.append(
            _audit_function(
                file_path=file_path,
                node=child,
                qualified_name=(
                    f"{node.name}.{child.name}"
                ),
                kind="method",
            )
        )

    return tuple(
        audits
    )

def _audit_module(
    file_path: Path,
) -> ModuleAudit:
    source = file_path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(
            file_path
        ),
    )

    definitions: list[
        DefinitionAudit
    ] = []

    for node in tree.body:
        if isinstance(
            node,
            ast.ClassDef,
        ):
            if _is_public(
                node.name
            ):
                definitions.extend(
                    _audit_class(
                        file_path=file_path,
                        node=node,
                    )
                )

            continue

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            if not _is_public(
                node.name
            ):
                continue

            definitions.append(
                _audit_function(
                    file_path=file_path,
                    node=node,
                    qualified_name=node.name,
                    kind="function",
                )
            )

    return ModuleAudit(
        file_path=file_path,
        has_docstring=(
            ast.get_docstring(tree)
            is not None
        ),
        definitions=tuple(
            definitions
        ),
    )

def _definition_start_line(
    node: (
        ast.ClassDef
        | ast.FunctionDef
        | ast.AsyncFunctionDef
    ),
) -> int:
    decorator_lines = tuple(
        decorator.lineno
        for decorator in node.decorator_list
    )

    if not decorator_lines:
        return node.lineno

    return min(
        node.lineno,
        *decorator_lines,
    )

def _blank_line_count(
    lines: list[str],
    previous_end: int,
    current_start: int,
) -> int:
    between = lines[
        previous_end:current_start - 1
    ]

    return sum(
        not line.strip()
        for line in between
    )

def _audit_import_spacing(
    *,
    file_path: Path,
    tree: ast.Module,
    lines: list[str],
) -> list[
    FormattingIssue
]:
    issues: list[
        FormattingIssue
    ] = []

    body_index = 0

    if (
        tree.body
        and isinstance(
            tree.body[0],
            ast.Expr,
        )
        and isinstance(
            tree.body[0].value,
            ast.Constant,
        )
        and isinstance(
            tree.body[0].value.value,
            str,
        )
    ):
        body_index = 1

    import_nodes: list[
        ast.Import
        | ast.ImportFrom
    ] = []

    while body_index < len(
        tree.body
    ):
        node = tree.body[
            body_index
        ]

        if not isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
            ),
        ):
            break

        import_nodes.append(
            node
        )

        body_index += 1

    for previous, current in zip(
        import_nodes,
        import_nodes[1:],
    ):
        blank_lines = _blank_line_count(
            lines,
            previous.end_lineno
            or previous.lineno,
            current.lineno,
        )

        if blank_lines != 0:
            issues.append(
                FormattingIssue(
                    file_path=file_path,
                    line_number=current.lineno,
                    message=(
                        "imports must not be "
                        "separated by blank lines"
                    ),
                )
            )

    if (
        import_nodes
        and body_index < len(
            tree.body
        )
    ):
        previous = import_nodes[-1]
        current = tree.body[
            body_index
        ]

        current_line = getattr(
            current,
            "lineno",
            None,
        )

        if current_line is not None:
            blank_lines = (
                _blank_line_count(
                    lines,
                    previous.end_lineno
                    or previous.lineno,
                    current_line,
                )
            )

            if blank_lines != 1:
                issues.append(
                    FormattingIssue(
                        file_path=file_path,
                        line_number=current_line,
                        message=(
                            "exactly one blank "
                            "line must follow "
                            "the import block"
                        ),
                    )
                )

    return issues

def _audit_top_level_spacing(
    *,
    file_path: Path,
    tree: ast.Module,
    lines: list[str],
) -> list[
    FormattingIssue
]:
    issues: list[
        FormattingIssue
    ] = []

    definition_types = (
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
    )

    for previous, current in zip(
        tree.body,
        tree.body[1:],
    ):
        if not isinstance(
            previous,
            definition_types,
        ):
            continue

        if not isinstance(
            current,
            definition_types,
        ):
            continue

        current_start = (
            _definition_start_line(
                current
            )
        )

        blank_lines = _blank_line_count(
            lines,
            previous.end_lineno
            or previous.lineno,
            current_start,
        )

        if blank_lines != 1:
            issues.append(
                FormattingIssue(
                    file_path=file_path,
                    line_number=current_start,
                    message=(
                        "exactly one blank line "
                        "must separate top-level "
                        "definitions"
                    ),
                )
            )

    return issues

def _audit_method_spacing(
    *,
    file_path: Path,
    tree: ast.Module,
    lines: list[str],
) -> list[
    FormattingIssue
]:
    issues: list[
        FormattingIssue
    ] = []

    method_types = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
    )

    for node in tree.body:
        if not isinstance(
            node,
            ast.ClassDef,
        ):
            continue

        for previous, current in zip(
            node.body,
            node.body[1:],
        ):
            if not isinstance(
                previous,
                method_types,
            ):
                continue

            if not isinstance(
                current,
                method_types,
            ):
                continue

            current_start = (
                _definition_start_line(
                    current
                )
            )

            blank_lines = (
                _blank_line_count(
                    lines,
                    previous.end_lineno
                    or previous.lineno,
                    current_start,
                )
            )

            if blank_lines != 1:
                issues.append(
                    FormattingIssue(
                        file_path=file_path,
                        line_number=current_start,
                        message=(
                            "exactly one blank "
                            "line must separate "
                            "class methods"
                        ),
                    )
                )

    return issues

def _audit_formatting(
    file_path: Path,
) -> tuple[
    FormattingIssue,
    ...,
]:
    source = file_path.read_text(
        encoding="utf-8"
    )

    lines = source.splitlines()

    tree = ast.parse(
        source,
        filename=str(
            file_path
        ),
    )

    issues = []

    issues.extend(
        _audit_import_spacing(
            file_path=file_path,
            tree=tree,
            lines=lines,
        )
    )

    issues.extend(
        _audit_top_level_spacing(
            file_path=file_path,
            tree=tree,
            lines=lines,
        )
    )

    issues.extend(
        _audit_method_spacing(
            file_path=file_path,
            tree=tree,
            lines=lines,
        )
    )

    return tuple(
        issues
    )

def _relative(
    path: Path,
) -> str:
    return str(
        path.relative_to(
            REPOSITORY_ROOT
        )
    )

def _percentage(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 100.0

    return round(
        numerator
        / denominator
        * 100.0,
        1,
    )

def _subsystem_name(
    path: Path,
) -> str:
    relative = path.relative_to(
        REPOSITORY_ROOT
    )

    parts = relative.parts

    if not parts:
        return "other"

    if parts[0] == "scripts":
        return "scripts"

    if parts[0] != "backend":
        return parts[0]

    if len(parts) == 2:
        return "behavior"

    return parts[1]

def _print_subsystem_summary(
    definitions: tuple[
        DefinitionAudit,
        ...,
    ],
) -> None:
    subsystem_names = sorted(
        {
            _subsystem_name(
                definition.file_path
            )
            for definition in definitions
        }
    )

    print(
        "=== PUBLIC API DOCUMENTATION BY SUBSYSTEM ==="
    )

    print()

    for subsystem in subsystem_names:
        subset = tuple(
            definition
            for definition in definitions
            if (
                _subsystem_name(
                    definition.file_path
                )
                == subsystem
            )
        )

        documented = sum(
            definition.has_docstring
            for definition in subset
        )

        missing = (
            len(subset)
            - documented
        )

        classes = sum(
            definition.kind == "class"
            for definition in subset
        )

        functions = sum(
            definition.kind == "function"
            for definition in subset
        )

        methods = sum(
            definition.kind == "method"
            for definition in subset
        )

        parameter_issues = sum(
            definition.parameters_annotated
            is False
            for definition in subset
        )

        return_issues = sum(
            definition.return_annotated
            is False
            for definition in subset
        )

        print(
            f"{subsystem}"
        )

        print(
            "  documented -> "
            f"{documented}/"
            f"{len(subset)} "
            f"({_percentage(documented, len(subset))}%)"
        )

        print(
            "  missing -> "
            f"{missing}"
        )

        print(
            "  classes -> "
            f"{classes}"
        )

        print(
            "  functions -> "
            f"{functions}"
        )

        print(
            "  methods -> "
            f"{methods}"
        )

        print(
            "  parameter type issues -> "
            f"{parameter_issues}"
        )

        print(
            "  return type issues -> "
            f"{return_issues}"
        )

        print()

def main() -> int:
    python_files = (
        _iter_python_files()
    )

    modules = tuple(
        _audit_module(path)
        for path in python_files
    )

    formatting_issues = tuple(
        issue
        for path in python_files
        for issue in _audit_formatting(
            path
        )
    )

    definitions = tuple(
        definition
        for module in modules
        for definition in module.definitions
    )

    documented_modules = sum(
        module.has_docstring
        for module in modules
    )

    documented_definitions = sum(
        definition.has_docstring
        for definition in definitions
    )

    typed_parameter_definitions = tuple(
        definition
        for definition in definitions
        if (
            definition.parameters_annotated
            is not None
        )
    )

    annotated_parameters = sum(
        bool(
            definition.parameters_annotated
        )
        for definition
        in typed_parameter_definitions
    )

    typed_return_definitions = tuple(
        definition
        for definition in definitions
        if (
            definition.return_annotated
            is not None
        )
    )

    annotated_returns = sum(
        bool(
            definition.return_annotated
        )
        for definition
        in typed_return_definitions
    )

    print(
        "=== RAVIN BACKEND DOCUMENTATION AUDIT ==="
    )

    print()

    print(
        "Python modules -> "
        f"{len(modules)}"
    )

    print(
        "Modules with docstrings -> "
        f"{documented_modules}/"
        f"{len(modules)} "
        f"({_percentage(documented_modules, len(modules))}%)"
    )

    print(
        "Public definitions -> "
        f"{len(definitions)}"
    )

    print(
        "Public definitions with docstrings -> "
        f"{documented_definitions}/"
        f"{len(definitions)} "
        f"({_percentage(documented_definitions, len(definitions))}%)"
    )

    print(
        "Public callables with all parameters typed -> "
        f"{annotated_parameters}/"
        f"{len(typed_parameter_definitions)} "
        f"({_percentage(annotated_parameters, len(typed_parameter_definitions))}%)"
    )

    print(
        "Public callables with return annotations -> "
        f"{annotated_returns}/"
        f"{len(typed_return_definitions)} "
        f"({_percentage(annotated_returns, len(typed_return_definitions))}%)"
    )

    print(
        "Formatting violations -> "
        f"{len(formatting_issues)}"
    )

    print()
    _print_subsystem_summary(
        definitions
    )

    print(
        "=== MISSING MODULE DOCSTRINGS ==="
    )

    missing_modules = tuple(
        module
        for module in modules
        if not module.has_docstring
    )

    if not missing_modules:
        print(
            "PASS - all audited modules have docstrings."
        )
    else:
        for module in missing_modules:
            print(
                _relative(
                    module.file_path
                )
            )

    print()
    print(
        "=== PUBLIC DEFINITION ISSUES ==="
    )

    issue_count = 0

    for definition in definitions:
        issues: list[str] = []

        if not definition.has_docstring:
            issues.append(
                "missing-docstring"
            )

        if (
            definition.parameters_annotated
            is False
        ):
            issues.append(
                "incomplete-parameter-types"
            )

        if (
            definition.return_annotated
            is False
        ):
            issues.append(
                "missing-return-type"
            )

        if not issues:
            continue

        issue_count += 1

        print(
            f"{_relative(definition.file_path)}"
            f" :: {definition.kind} "
            f"{definition.qualified_name}"
            f" :: {', '.join(issues)}"
        )

    if issue_count == 0:
        print(
            "PASS - no public definition issues found."
        )

    print()
    print(
        "=== RAVIN FORMATTING ISSUES ==="
    )

    if not formatting_issues:
        print(
            "PASS - all audited files match "
            "RAVIN spacing rules."
        )
    else:
        for issue in formatting_issues:
            print(
                f"{_relative(issue.file_path)}"
                f":{issue.line_number}"
                f" :: {issue.message}"
            )

    print()
    print(
        "Definitions requiring attention -> "
        f"{issue_count}"
    )

    print(
        "Formatting violations -> "
        f"{len(formatting_issues)}"
    )

    print()
    print(
        "Audit mode -> READ ONLY"
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(
        main()
    )