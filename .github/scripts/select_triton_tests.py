#!/usr/bin/env python


# Imports
# ------------------------------------------------------------------------------

# Python standard library.
import argparse
import ast
import functools
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Third party libraries.
import networkx as nx

logger = logging.getLogger(__name__)

# Small utility functions.
# ------------------------------------------------------------------------------


def log_file_list(log_level: int, files: list[Path]) -> None:
    if not logging.getLogger().isEnabledFor(log_level):
        return
    for f in files:
        logger.log(log_level, "* %s", f)


# Structure of Triton source files.
# ------------------------------------------------------------------------------


def check_dir(p: Path) -> Path:
    if not p.exists():
        logger.critical("Required directory [%s] doesn't exist.", p)
        sys.exit(1)
    if not p.is_dir():
        logger.critical("Required directory [%s] isn't a directory.", p)
        sys.exit(1)
    return p


def optional_dir(p: Path) -> Path | None:
    """
    Like `check_dir`, but for directories that are allowed to be missing, e.g.
    legacy layouts that are being migrated away. Returns `None` instead of
    exiting so that callers can simply contribute no candidates.
    """
    if not p.exists():
        logger.debug("Optional directory [%s] doesn't exist.", p)
        return None
    if not p.is_dir():
        logger.warning("Optional directory [%s] isn't a directory.", p)
        return None
    return p


@functools.cache
def root_dir() -> Path:
    return check_dir(Path(__file__).parent.parent.parent)


@functools.cache
def triton_op_dir() -> Path:
    return check_dir(root_dir() / "aiter" / "ops" / "triton")


@functools.cache
def triton_config_dir() -> Path:
    return check_dir(triton_op_dir() / "configs")


@functools.cache
def triton_gemm_config_dir() -> Path | None:
    # Legacy flat GEMM config layout. It shrinks and eventually disappears as
    # config families migrate to `configs/<arch>/<backend>/gemm/<d_type>/`, so
    # its absence isn't an error: there are simply no legacy files left to match.
    return optional_dir(triton_config_dir() / "gemm")


def list_files(dir: Path, suffix: str = "") -> set[Path]:
    return {p.relative_to(root_dir()) for p in dir.glob(f"**/*{suffix}") if p.is_file()}


def list_triton_op_files() -> set[Path]:
    files = list_files(triton_op_dir(), suffix=".py")
    logger.debug("Found %d Triton operator source files.", len(files))
    return files


def list_triton_kernel_files(kernel_dir: Path) -> set[Path]:
    files = list_files(kernel_dir, suffix=".py")
    logger.debug("Found %d Triton kernel source files.", len(files))
    return files


def list_triton_config_files() -> set[Path]:
    files = list_files(triton_config_dir(), suffix=".json")
    logger.debug("Found %d Triton kernel config files.", len(files))
    return files


def list_triton_test_files(test_dir: Path) -> set[Path]:
    files = list_files(test_dir, suffix=".py")
    logger.debug("Found %d Triton test source files.", len(files))
    return files


def list_triton_bench_files(bench_dir: Path) -> set[Path]:
    files = list_files(bench_dir, suffix=".py")
    logger.debug("Found %d Triton benchmark source files.", len(files))
    return files


def list_triton_source_files() -> (
    tuple[set[Path], list[Path], list[Path], list[Path], list[Path]]
):
    kernels_dir = check_dir(triton_op_dir() / "_triton_kernels")
    op_test_dir = check_dir(root_dir() / "op_tests")
    test_dir = check_dir(op_test_dir / "triton_tests")
    bench_dir = check_dir(op_test_dir / "op_benchmarks" / "triton")
    op_files = list_triton_op_files()
    kernel_files = list_triton_kernel_files(kernels_dir)
    config_files = list_triton_config_files()
    test_files = list_triton_test_files(test_dir)
    bench_files = list_triton_bench_files(bench_dir)
    all_files = op_files | kernel_files | config_files | test_files | bench_files
    return (
        all_files,
        sorted(kernel_files),
        sorted(config_files),
        sorted(test_files),
        sorted(bench_files),
    )


# Matching of kernel config files.
# ------------------------------------------------------------------------------


# Architectures used as the `<arch>-` filename prefix of the legacy flat config
# layout. In the nested layout the architecture is a directory instead, and it's
# matched with a wildcard, see `Visitor.add_config_dir`. Keep this in sync with
# the architecture table in `aiter/ops/triton/configs/CLAUDE.md`.
DEVICES: frozenset[str] = frozenset(
    ["gfx942", "gfx950", "gfx1151", "gfx1200", "gfx1201", "gfx1250"]
)

# Single letter placeholders for JSON config file template strings. You can add
# more letters if necessary.
MNK_PLACEHOLDERS: str = "MNK"
NUM_PATTERN: str = r"\d+"

MOE_DTYPES: tuple[str, ...] = (
    "DEFAULT",
    "FP8_W8A8",
    "INT8_W8A16",
    "INT8_W8A8",
    "INT4_W4A16",
    "MX_FP4",
)


def fold_config_name(config_name: str) -> str:
    """
    Folds a config name into the `<d_type>` directory name of the nested config
    layout, the same way `gemm_config_utils._dtype_dir()` does it, e.g.
    `GEMM-AFP4WFP4` becomes `gemm_afp4wfp4`.

    :param config_name: Config family name, e.g. `GEMM-AFP4WFP4`. May contain
                        `{...}` interpolation placeholders, which are left
                        untouched so that later expansion steps still recognize
                        them.
    :return: Directory name of the config family in the nested config layout.
    """
    return "".join(
        part if part.startswith("{") else part.lower().replace("-", "_")
        for part in re.split(r"(\{[^{}]*\})", config_name)
    )


def expand_mnk(json_string: str, config_files: list[Path]) -> list[str]:
    """
    Expands template strings containing M/N/K placeholders by matching them
    against actual config file paths.

    :param json_string: Template string that references kernel config JSON files.
                        May contain M/N/K placeholders to be expanded.
    :param config_files: All Triton kernel config JSON files available in the
                         filesystem.
    :return: List of template strings with expanded M/N/K placeholders or the
             a list containing just the input template string as-is if there
             are no M/N/K placeholders to be expanded.
    """
    # Early exit if no placeholders present.
    if not any(f"{{{p}}}" in json_string for p in MNK_PLACEHOLDERS):
        logger.debug("No M/N/K placeholders in [%s].", json_string)
        return [json_string]
    # Strip `f"` prefix and `"` suffix, then escape special regex characters. For
    # example, `f"gfx950-GEMM-N={N}-K={K}.json"` becomes `gfx950-GEMM-N=\{N\}-K=\{K\}.json`.
    pattern_str = re.escape(json_string[2:-1])
    # Replace each escaped placeholder with a named capture regex group.
    for p in MNK_PLACEHOLDERS:  # `p` will be `M`, then `N`, then `K`.
        # For instance, if `p` is `M` then `\{M\}` will be replaced by `(?P<M>\d+)`.
        # Following the example:
        # 1. there's no M, pattern is not changed.
        # 2. `gfx950-GEMM-N=\{N\}-K=\{K\}.json` becomes `gfx950-GEMM-N=(?P<N>\d+)-K=\{K\}.json`
        # 3. `gfx950-GEMM-N=(?P<N>\d+)-K=\{K\}.json` becomes `gfx950-GEMM-N=(?P<N>\d+)-K=(?P<K>\d+).json`
        pattern_str = pattern_str.replace(
            rf"\{{{p}\}}",
            rf"(?P<{p}>{NUM_PATTERN})",
        )
    # Compile regex with anchors to match entire path.
    pattern = re.compile(f"^{pattern_str}$")
    logger.debug("M/N/K regex is [%s].", pattern.pattern)
    # Return f-string representations of matching config paths.
    return [f'f"{path}"' for c in config_files if pattern.match(path := c.as_posix())]


def expand_globs(json_string: str, config_files: list[Path]) -> list[str]:
    """
    Expands template strings containing `*` wildcards by matching them against
    actual config file paths. Like a shell glob, a wildcard matches anything
    within a single path component.

    :param json_string: Template string that references kernel config JSON files.
                        May contain `*` wildcards to be expanded.
    :param config_files: All Triton kernel config JSON files available in the
                         filesystem.
    :return: List of template strings with expanded wildcards or a list
             containing just the input template string as-is if there are no
             wildcards to be expanded.
    """
    # Early exit if no wildcards present.
    if "*" not in json_string:
        logger.debug("No wildcards in [%s].", json_string)
        return [json_string]
    # Strip `f"` prefix and `"` suffix, escape special regex characters, then turn
    # every escaped wildcard into a group matching a single path component. For
    # example, `f"configs/*/*/moe/a8w4/*.json"` becomes
    # `configs/[^/]*/[^/]*/moe/a8w4/[^/]*\.json`.
    pattern_str = re.escape(json_string[2:-1]).replace(re.escape("*"), "[^/]*")
    # Compile regex with anchors to match entire path.
    pattern = re.compile(f"^{pattern_str}$")
    logger.debug("Wildcard regex is [%s].", pattern.pattern)
    # Return f-string representations of matching config paths.
    return [f'f"{path}"' for c in config_files if pattern.match(path := c.as_posix())]


def expand_moe_dtypes(json_strings: list[str]) -> list[str]:
    expanded_moe_dtypes: list[str] = []
    for s in json_strings:
        # Legacy flat file names keep the config name verbatim, e.g.
        # `gfx950-MOE-FP8_W8A8.json`, while the `<d_type>` directory of the nested
        # layout folds it with `fold_config_name`, e.g. `moe_fp8_w8a8`.
        if r"MOE-{dtype_str}" in s:
            dtypes = MOE_DTYPES
        elif r"moe_{dtype_str}" in s:
            dtypes = tuple(dtype.lower() for dtype in MOE_DTYPES)
        else:
            expanded_moe_dtypes.append(s)
            continue
        expanded_moe_dtypes.extend(s.replace(r"{dtype_str}", dtype) for dtype in dtypes)
    return expanded_moe_dtypes


def expand_interpolations(json_string: str, config_files: list[Path]) -> list[str]:
    if not (json_string.startswith(("f'", 'f"'))):
        return [json_string]
    # Replace config path placeholder
    if r"{AITER_TRITON_CONFIGS_PATH}" in json_string:
        json_string = json_string.replace(
            r"{AITER_TRITON_CONFIGS_PATH}",
            str(triton_config_dir().relative_to(root_dir()).as_posix()),
        )
        logger.debug("Resolved {AITER_TRITON_CONFIGS_PATH}: [%s]", json_string)
    # Expand device variants
    expanded = [json_string]
    if r"{dev}" in json_string:
        expanded = [s.replace(r"{dev}", dev) for s in expanded for dev in DEVICES]
        logger.debug("Resolved {dev}: %s", expanded)
    # Expand GEMM M-N-K patterns
    expanded = [
        expanded_mnk for s in expanded for expanded_mnk in expand_mnk(s, config_files)
    ]
    # Expand MOE data type variants
    expanded = expand_moe_dtypes(expanded)
    # Expand nested config layout wildcards
    expanded = [
        expanded_glob
        for s in expanded
        for expanded_glob in expand_globs(s, config_files)
    ]
    # Clean up f-string delimiters if no more interpolation needed
    expanded = [s[2:-1] if not any(c in s for c in "{}") else s for s in expanded]
    return expanded


def resolve_path(json_string: str) -> Path | None:
    p = Path(json_string)
    # Try absolute path
    if p.is_absolute() and p.exists() and p.is_file():
        return p.relative_to(root_dir())
    # Try relative to root
    p = root_dir() / json_string
    if p.exists() and p.is_file():
        return p.relative_to(root_dir())
    return None


def resolve_json_strings(
    json_strings: list[str], config_files: list[Path]
) -> list[Path]:
    # Expand all interpolations first
    expanded_strings = [
        expanded
        for json_string in json_strings
        for expanded in expand_interpolations(json_string, config_files)
    ]
    # Sort and deduplicate
    expanded_strings = sorted(set(expanded_strings))
    # Resolve to actual paths
    resolved: list[Path] = []
    unresolved: list[str] = []
    for json_string in expanded_strings:
        if resolved_path := resolve_path(json_string):
            resolved.append(resolved_path)
        else:
            unresolved.append(json_string)
    # Log results
    if resolved:
        logger.debug("Resolved JSON config files:")
        log_file_list(logging.DEBUG, resolved)
    log_level = logging.DEBUG
    if unresolved and logging.getLogger().isEnabledFor(log_level):
        logger.log(log_level, "Unresolved JSON strings:")
        for s in unresolved:
            logger.log(log_level, "* %s", s)
    return resolved


def resolve_gemm_config_names(gemm_config_names: list[str]) -> list[Path]:
    # Legacy flat layout only. Config families that already moved to the nested
    # layout are matched from their `resolve_config_dir` call sites instead, see
    # `Visitor.add_config_dir`.
    gemm_config_dir = triton_gemm_config_dir()
    if not gemm_config_names or gemm_config_dir is None:
        return []
    gemm_configs = [
        p.relative_to(root_dir())
        for dev in DEVICES
        for gemm_config_name in gemm_config_names
        for p in gemm_config_dir.glob(f"{dev}-{gemm_config_name}*.json")
        if p.is_file()
    ]
    if gemm_configs:
        logger.debug("Resolved JSON GEMM config files:")
        log_file_list(logging.DEBUG, gemm_configs)
    return gemm_configs


def resolve_configs(
    json_strings: list[str],
    gemm_config_names: list[str],
    config_files: list[Path],
) -> list[Path]:
    return sorted(
        set(resolve_json_strings(json_strings, config_files)).union(
            resolve_gemm_config_names(gemm_config_names)
        )
    )


# Git commands.
# ------------------------------------------------------------------------------


def git(args: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git"] + shlex.split(args),
            capture_output=True,
            text=True,
            check=check,
            cwd=root_dir(),  # always run git commands from repo root
        )
    except FileNotFoundError:
        logger.critical("Git not found.")
        sys.exit(1)
    except subprocess.CalledProcessError:
        logger.critical("Malformed Git command: [git %s].", args)
        sys.exit(1)


def git_current_branch() -> str:
    return git("rev-parse --abbrev-ref HEAD").stdout.rstrip()


def git_check_branch(branch: str) -> None:
    if git(f"rev-parse --verify --quiet {branch}", check=False).returncode != 0:
        logger.critical("Branch [%s] doesn't exist.", branch)
        sys.exit(1)


def git_filename_diff(source_branch: str, target_branch: str) -> set[Path]:
    diff_output = git(
        f"diff --name-only {target_branch} {source_branch}"
    ).stdout.rstrip()
    if not diff_output:
        return set()
    files = set()
    for diff_p in diff_output.splitlines():
        abs_path = root_dir() / diff_p
        if abs_path.exists() and abs_path.is_file():
            # Add path relative to root_dir() - this will match paths from `list_files` function.
            files.add(abs_path.relative_to(root_dir()))
    logger.debug(
        "There %s %d file%s in the diff from [%s] to [%s].",
        "is" if len(files) == 1 else "are",
        len(files),
        "" if len(files) == 1 else "s",
        source_branch,
        target_branch,
    )
    return files


def get_filename_diff(source_branch: str | None, target_branch: str) -> set[Path]:
    if source_branch is None:
        source_branch = git_current_branch()
        logger.info(
            "Source branch wasn't provided, using current branch [%s] as source branch.",
            source_branch,
        )
    else:
        git_check_branch(source_branch)

    git_check_branch(target_branch)

    if target_branch == source_branch:
        logger.error("Source and target branches must be different.")
        sys.exit(1)

    return git_filename_diff(source_branch, target_branch)


# Source file parsing.
# ------------------------------------------------------------------------------


def join_fstring(node: ast.JoinedStr) -> str:
    """
    Flattens an f-string AST node into a plain string, keeping every
    interpolation as a `{expr}` placeholder for later expansion.
    """
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            # Unparse the inner expression to a readable form.
            parts.append(f"{{{ast.unparse(value.value)}}}")
    return "".join(parts)


def literal_str(node: ast.expr) -> str | None:
    """
    Extracts the value of a string literal AST node: plain strings verbatim,
    f-strings with their interpolations kept as `{expr}` placeholders. Returns
    `None` for anything that isn't a string literal.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return join_fstring(node)
    return None


class Visitor(ast.NodeVisitor):
    imports_to_ignore: frozenset[str] = frozenset(
        [
            "einops",
            "iris",
            "jax",
            "jinja2",
            "matplotlib",
            "mori",
            "numpy",
            "packaging",
            "pandas",
            "prettytable",
            "psutil",
            "pybind11",
            "pytest",
            "setuptools",
            "torch",
            "triton",
            "vllm",
            "yaml",
            "zmq",
        ]
    )
    json_strings_to_ignore: frozenset[str] = frozenset(
        [
            "./utils/model_configs.json",
            ".json",
            "empty_kernel.json",
            "f'{kernel_name}.json'",
            "utils/model_configs.json",
        ]
    )

    @classmethod
    def is_import_of_interest(cls, import_: str) -> bool:
        return (
            import_ not in sys.stdlib_module_names
            and import_ not in cls.imports_to_ignore
            and not any(
                import_.startswith(module + ".") for module in cls.imports_to_ignore
            )
            and not any(
                import_.startswith(module + ".") for module in sys.stdlib_module_names
            )
        )

    @classmethod
    def is_json_string_of_interest(cls, json_string: str) -> bool:
        return json_string not in cls.json_strings_to_ignore

    def __init__(self, source_file: Path) -> None:
        # Remove extension from source file, and split directories into module parts.
        self.source_file: Path = source_file
        self.source_module_parts: tuple[str, ...] = self.source_file.with_suffix(
            ""
        ).parts
        self.dependencies: set[Path] = set()
        self.json_strings: set[str] = set()
        self.gemm_config_names: set[str] = set()

    def add_dependency(self, import_: str, suppress_warning: bool = False) -> None:
        if not import_ or not self.__class__.is_import_of_interest(import_):
            return
        import_py_file = import_.replace(".", os.sep) + ".py"
        # Check if it's a module file (e.g., `foo/bar.py`).
        p = root_dir() / import_py_file
        if p.exists() and p.is_file():
            self.dependencies.add(p.relative_to(root_dir()))
            return
        # Check if it's a package directory with `__init__.py`.
        p_dir = (root_dir() / import_py_file).with_suffix("")
        p_init = p_dir / "__init__.py"
        if p_init.exists() and p_init.is_file():
            # Add dependency as the `__init__.py` file of the package.
            self.dependencies.add(p_init.relative_to(root_dir()))
            return
        # Check for relative imports within the same package.
        p = root_dir() / self.source_file.parent / import_py_file
        if p.exists() and p.is_file():
            self.dependencies.add(p.relative_to(root_dir()))
            return
        # Check for relative package imports.
        p_dir = (root_dir() / self.source_file.parent / import_py_file).with_suffix("")
        p_init = p_dir / "__init__.py"
        if p_init.exists() and p_init.is_file():
            self.dependencies.add(p_init.relative_to(root_dir()))
            return
        if not suppress_warning:
            # Check if it's a directory without `__init__.py`` (namespace package or external)
            p_dir = (root_dir() / import_py_file).with_suffix("")
            if p_dir.exists() and p_dir.is_dir():
                logger.debug(
                    "Directory [%s] exists but has no '__init__.py', skipping dependency [%s] of [%s].",
                    p_dir.relative_to(root_dir()),
                    import_,
                    self.source_file,
                )
            else:
                logger.warning(
                    "Unable to find [%s] dependency of [%s] on filesystem.",
                    import_,
                    self.source_file,
                )

    def add_json_string(self, json_string: str) -> None:
        if not json_string or not self.__class__.is_json_string_of_interest(
            json_string
        ):
            return
        self.json_strings.add(json_string)

    def add_config_dir(self, op: str, config_name: str) -> None:
        """
        Records the config directory that a `resolve_config_dir(op, config_name)`
        call site resolves to in the nested config layout,
        `configs/<arch>/<backend>/<op>/<d_type>/`, as a wildcard template.

        Loaders read `DEFAULT.json` and the specialized files of the family from
        that directory through `f"{cfg_dir}/..."` strings, which can't be resolved
        on their own. Matching the whole directory instead keeps the edge from
        every config file of the family to the code that reads it.
        """
        if not op or not config_name:
            return
        # Architecture and backend are directories here, so they're wildcards: all
        # of a family's config files feed the same modules regardless of which
        # architecture or backend they were tuned for.
        config_dir_glob = (
            r"{AITER_TRITON_CONFIGS_PATH}"
            f"/*/*/{op}/{fold_config_name(config_name)}/*.json"
        )
        self.add_json_string(f"f{config_dir_glob!r}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.add_dependency(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module_name = node.module if node.module else ""
        if node.level > 0:
            # Resolve absolute import from relative import.
            full_module_name = ".".join(self.source_module_parts[: -node.level])
            if module_name:
                full_module_name += "." + module_name
        else:
            full_module_name = module_name
        self.add_dependency(full_module_name)
        # For "from X import Y", also try to add Y as a potential module
        # (in case Y is a submodule rather than a name defined in X).
        if full_module_name and node.names[0].name != "*":
            for alias in node.names:
                potential_submodule = f"{full_module_name}.{alias.name}"
                self.add_dependency(potential_submodule, suppress_warning=True)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # Plain string literals.
        if isinstance(node.value, str) and node.value.lower().endswith(".json"):
            self.add_json_string(node.value)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        # f-strings.
        joined = join_fstring(node)
        if joined.lower().endswith(".json"):
            self.add_json_string(f"f{joined!r}")

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        func_name = ""
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        # `get_gemm_config(config_name, ...)` finds its config directory through
        # `resolve_config_dir("gemm", config_name, ...)`; direct call sites of the
        # shared probe name the operation themselves.
        op: str | None = None
        config_name: str | None = None
        if func_name == "get_gemm_config" and node.args:
            op, config_name = "gemm", literal_str(node.args[0])
        elif func_name == "resolve_config_dir" and len(node.args) > 1:
            op, config_name = literal_str(node.args[0]), literal_str(node.args[1])
        if op and config_name:
            # Nested layout, `configs/<arch>/<backend>/<op>/<d_type>/*.json`.
            self.add_config_dir(op, config_name)
            if op == "gemm":
                # Legacy flat layout, `configs/gemm/<arch>-<CONFIG_NAME>*.json`,
                # still holds every family that hasn't been migrated yet.
                self.gemm_config_names.add(config_name)
        self.generic_visit(node)


def parse_source_file(source_file: Path) -> tuple[list[Path], list[str], list[str]]:
    try:
        logger.debug("Parsing source file [%s]...", str(source_file))
        source = (root_dir() / source_file).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_file))
    except Exception:
        logger.exception("Skipping source file [%s].", source_file)
        return [], [], []

    visitor = Visitor(source_file)
    visitor.visit(tree)

    dependecies = sorted(visitor.dependencies)
    if not dependecies:
        logger.debug("No dependecies of interest in [%s].", source_file)
    else:
        logger.debug("Dependecies of interest in [%s]:", source_file)
        log_file_list(logging.DEBUG, dependecies)

    json_strings = sorted(visitor.json_strings)
    if not json_strings:
        logger.debug("No JSON strings in [%s].", source_file)
    else:
        logger.debug("JSON strings in [%s]: %s", source_file, str(json_strings))

    gemm_config_names = sorted(visitor.gemm_config_names)
    if not gemm_config_names:
        logger.debug("No GEMM config names in [%s].", source_file)
    else:
        logger.debug(
            "GEMM config names in [%s]: %s", source_file, str(gemm_config_names)
        )

    return dependecies, json_strings, gemm_config_names


def parse_source_file_recursively(
    graph: nx.DiGraph,
    source_file: Path,
    config_files: list[Path],
    visited: set[Path],
    deps_to_ignore: set[Path] | None = None,
) -> None:
    if deps_to_ignore is None:
        deps_to_ignore = set()
    stack = [source_file]

    while stack:
        current = stack.pop()
        if current in visited:
            continue

        dependencies, json_strings, gemm_config_names = parse_source_file(current)
        configs = resolve_configs(json_strings, gemm_config_names, config_files)

        # Add current node to the graph.
        current_str = str(current)
        graph.add_node(current_str)
        logger.debug("Added graph node [%s].", current_str)

        # Add dependencies of current node, and respective edges, to the graph.
        for d in dependencies:
            d_str = str(d)
            graph.add_node(d_str)
            logger.debug("Added graph node [%s].", d_str)
            graph.add_edge(d_str, current_str)
            logger.debug("Added graph edge [%s]->[%s].", d_str, current_str)

        # Add configs of current node, and respective edges, to the graph.
        for c in configs:
            c_str = str(c)
            graph.add_node(c_str)
            logger.debug("Added graph node [%s].", c_str)
            graph.add_edge(c_str, current_str)
            logger.debug("Added graph edge [%s]->[%s].", c_str, current_str)

        stack.extend(
            d
            for d in dependencies
            if (rd := root_dir() / d).is_file() and rd not in deps_to_ignore
        )
        visited.add(current)


# Dependency graph.
# ------------------------------------------------------------------------------


def tag_node(graph: nx.DiGraph, file: Path, tag: str) -> None:
    file_str = str(file)
    if file_str in graph.nodes:
        graph.nodes[file_str]["type"] = tag
        logger.debug("Tagged file [%s] as a '%s' in the graph.", file_str, tag)
    else:
        logger.warning(
            "Couldn't find file [%s] in the graph, unable to tag it as '%s'.",
            file_str,
            tag,
        )


def add_files_to_dependency_graph(
    graph: nx.DiGraph,
    files: list[Path],
    file_type: str,
    config_files: list[Path],
    visited: set[Path],
    deps_to_ignore: set[Path] | None = None,
) -> None:
    if deps_to_ignore is None:
        deps_to_ignore = set()
    for f in files:
        parse_source_file_recursively(
            graph, f, config_files, visited, deps_to_ignore=deps_to_ignore
        )
        tag_node(graph, f, file_type)


def build_dependency_graph(
    kernel_files: list[Path],
    config_files: list[Path],
    test_files: list[Path],
    bench_files: list[Path],
) -> nx.DiGraph:
    graph: nx.DiGraph = nx.DiGraph()
    visited: set[Path] = set()
    # These source files have dependencies that are hard to track and can be safely ignored for
    # Triton test selection purposes:
    deps_to_ignore: set[Path] = {
        root_dir() / "aiter" / "jit" / "core.py",
        root_dir() / "aiter" / "dist" / "utils.py",
        root_dir() / "csrc" / "cpp_itfs" / "hsaco_launcher.py",
    }
    assert all(
        d.is_file() and d.exists() for d in deps_to_ignore
    ), "All ignored source file dependencies must exist in the filesystem."
    # Add files that tests depends on.
    add_files_to_dependency_graph(
        graph, test_files, "test", config_files, visited, deps_to_ignore=deps_to_ignore
    )
    # Add files that benchmarks depends on.
    add_files_to_dependency_graph(
        graph,
        bench_files,
        "bench",
        config_files,
        visited,
        deps_to_ignore=deps_to_ignore,
    )
    logger.debug(
        "Built dependency graph of Triton source files with %d nodes and %d edges.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    # Tag kernel files.
    for kernel_file in kernel_files:
        if kernel_file.name != "__init__.py":
            tag_node(graph, kernel_file, "kernel")
    # Tag config files.
    for config_file in config_files:
        tag_node(graph, config_file, "config")
    return graph


def find_tests_to_run(graph: nx.DiGraph, diff_inter_triton: list[Path]) -> list[Path]:
    # Filter out unreachable files, i.e. Triton source files that are in the diff content but
    # aren't in the dependency graph.
    reachable_diff_inter_triton: list[Path] = []
    for p in diff_inter_triton:
        p_str = str(p)
        if p_str in graph:
            reachable_diff_inter_triton.append(p)
        else:
            logger.warning(
                "Triton source file [%s] isn't in the dependency graph, it's unreachable.",
                p_str,
            )
    if not reachable_diff_inter_triton:
        logger.warning("There are no reachable tests from the Triton diff.")
        logger.warning(
            "Please check test selection script, there might be a bug in it."
        )
        logger.warning(
            "Please check Triton code base, there may be some filesystem organizations that aren't taken into account."
        )
        return []

    # Figure out if all reachable files are benchmarks. If this is the case, then we should use
    # other graph traversal approach.
    is_all_bench: bool = all(
        graph.nodes[str(p)].get("type") == "bench" for p in reachable_diff_inter_triton
    )

    # Traverse the dependency graph, searching from tests that are reachable from the diff content.
    tests_to_run: set[Path] = set()
    for p in reachable_diff_inter_triton:
        p_str = str(p)
        logger.debug("Searching for tests related to Triton source file [%s]...", p_str)
        if not is_all_bench:
            # Forward traversal for non-benchmarks.
            reachable_files = nx.descendants(graph, p_str) | {p_str}
        else:
            # Backward traversal for benchmarks, filtering only dependencies on kernel files. After
            # that, perform a forward traversal from kernels. This strategy isn't perfect but it's
            # an attempt to take benchmark files into account. It may fail if a benchmark utility is
            # changed in isolation.
            reachable_files = {
                kernel_descendant
                for p_ancestor in nx.ancestors(graph, p_str)
                if graph.nodes[p_ancestor].get("type") == "kernel"
                for kernel_descendant in nx.descendants(graph, p_ancestor)
            }
        logger.debug(
            "There %s %d file%s reachable from [%s].",
            "is" if len(reachable_files) == 1 else "are",
            len(reachable_files),
            "" if len(reachable_files) == 1 else "s",
            p_str,
        )

        test_files = {
            Path(f) for f in reachable_files if graph.nodes[f].get("type") == "test"
        }
        if test_files:
            logger.debug(
                "There %s %d test%s reachable from [%s].",
                "is" if len(test_files) == 1 else "are",
                len(test_files),
                "" if len(test_files) == 1 else "s",
                p_str,
            )
            tests_to_run.update(test_files)
        else:
            logger.warning(
                "Couldn't find test files related to [%s] Triton source.", p_str
            )

    if tests_to_run:
        sorted_tests_to_run = sorted(tests_to_run)
        logger.info(
            "There %s %d test%s reachable from the Triton diff:",
            "is" if len(sorted_tests_to_run) == 1 else "are",
            len(sorted_tests_to_run),
            "" if len(sorted_tests_to_run) == 1 else "s",
        )
        log_file_list(logging.INFO, sorted_tests_to_run)
        return sorted_tests_to_run
    else:
        logger.warning("Couldn't find any test file related to Triton diff.")
        logger.warning(
            "Please check test selection script, there might be a bug in it."
        )
        logger.warning("Please check Triton code base, there may be untested kernels.")
        return []


# Writing to GitHub environment file.
# ------------------------------------------------------------------------------


def write_env_file(env_var: str, env_file: str, tests_to_run: list[Path]) -> None:
    if env_var is None or not (env_var := env_var.strip()):
        logger.info(
            "Environment variable is absent, environment file won't be written."
        )
        return
    if env_file is None or not (env_file := env_file.strip()):
        logger.info("Environment file is absent, it won't be written.")
        return
    if not tests_to_run:
        logger.warning(
            "List of tests to run is empty, enviroment file won't be written."
        )
        return
    tests_to_run_joined_str = " ".join(str(t) for t in tests_to_run)
    env_file_data = f"{env_var}={tests_to_run_joined_str}"
    logger.debug("Writing [%s] to [%s]...", env_file_data, env_file)
    try:
        with open(env_file, "a") as env_file_fd:
            env_file_fd.write(env_file_data + "\n")
        logger.info("Wrote tests to run to [%s] environment file.", env_file)
    except OSError:
        logger.exception("I/O error while writing to [%s] environment file.", env_file)
        logger.info("The entire Triton test suite will be executed.")


# Command line interface parsing.
# ------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="select which Triton tests to run based on git diff"
    )
    parser.add_argument(
        "-s", "--source", help="source branch, defaults to current branch"
    )
    parser.add_argument(
        "-t", "--target", default="main", help="target branch, defaults to main"
    )
    parser.add_argument(
        "-v",
        "--env-var",
        default="TRITON_TEST",
        help="environment variable to store which tests to run, defaults to TRITON_TEST",
    )
    parser.add_argument(
        "-f",
        "--env-file",
        help="environment file to write, won't write anything if absent",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str.lower,
        choices=["critical", "error", "warning", "info", "debug", "off"],
        default="info",
        help="log level to enable (default: info)",
    )
    args = parser.parse_args()
    args.log_level = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "off": logging.CRITICAL + 1000,
    }[args.log_level]
    return args


# Main logic and script entry point.
# ------------------------------------------------------------------------------


def main_logic(args: argparse.Namespace) -> None:
    diff_files = get_filename_diff(args.source, args.target)
    all_files, kernel_files, config_files, test_files, bench_files = (
        list_triton_source_files()
    )
    diff_inter_triton = diff_files & all_files
    del diff_files, all_files

    if not diff_inter_triton:
        logger.info(
            "There are no Triton source files in diff, there's no need to run Triton tests."
        )
        return

    logger.info(
        "There %s %d Triton source file%s in the diff:",
        "is" if len(diff_inter_triton) == 1 else "are",
        len(diff_inter_triton),
        "" if len(diff_inter_triton) == 1 else "s",
    )

    sorted_diff_inter_triton = sorted(diff_inter_triton)
    del diff_inter_triton
    log_file_list(logging.INFO, sorted_diff_inter_triton)

    graph = build_dependency_graph(kernel_files, config_files, test_files, bench_files)
    tests_to_run = find_tests_to_run(graph, sorted_diff_inter_triton)

    if not tests_to_run:
        return

    write_env_file(args.env_var, args.env_file, tests_to_run)


def main() -> None:
    start_timestamp = time.perf_counter()
    args = parse_args()
    logging.basicConfig(format="%(levelname)s|%(message)s", level=args.log_level)
    main_logic(args)
    end_timestamp = time.perf_counter()
    elapsed_time_s = end_timestamp - start_timestamp
    logger.info("Finished, execution took %.2f seconds.", elapsed_time_s)


if __name__ == "__main__":
    main()
