"""Classify run-level errors from driver output JSONs into a structured CSV,
and optionally summarise results into pivot tables and top-K breakdowns.

One row per run attempt (including successes). Matchers are selected per
(language, parallelism_model) pair: GENERIC_MATCHERS (language-agnostic outcomes)
apply to every pair, and each pair's own content matchers live in MATCHER_SETS.
Currently only ('python', 'pycompss') is populated; other pairs fall back to the
generic outcomes plus a catch-all. To support another pair (e.g. ('cpp', 'cuda')),
write its matchers and register them under that key. First match wins.

Usage:
    python classify-errors.py kernel-20
    python classify-errors.py kernel-20 --include-success
    python classify-errors.py kernel-20 --summarize
    python classify-errors.py kernel-20 --only-summarize   # skip re-classification
    python classify-errors.py drivers/outputs/kernel-20    # full path also accepted
"""
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd

# ── Text preprocessing ────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

# Lines that are COMPSs / JVM boilerplate and carry no diagnostic value.
_BOILERPLATE = re.compile(
    r'^(Picked up JAVA_TOOL_OPTIONS'
    r'|WARNING: Runtime environment or build system'
    r'|WARNING: COMPSs Properties file is null'
    r'|WARNING: Unexpected argument'       # noisy but separate category below
    r'|\[INFO\]'
    r'|\[DEBUG\]'
    r'|$)',
    re.MULTILINE,
)

_BOILERPLATE_LINE = re.compile(
    r'^(\s*Picked up JAVA_TOOL_OPTIONS'
    r'|\s*WARNING: Runtime environment or build system'
    r'|\s*WARNING: COMPSs Properties file is null'
    r'|\s*WARNING: Unexpected argument'
    r'|\s*\[INFO\]'
    r'|\s*\[DEBUG\]'
    r'|\s*$)',
)

def _clean(stderr: str) -> str:
    """Strip ANSI codes."""
    return _ANSI_RE.sub('', stderr or '')

def _stderr_snippet(stderr: str, n_lines: int = 20) -> str:
    """Return the last n_lines meaningful lines of stderr (ANSI stripped, boilerplate removed).

    Strips the [Timeout] prefix and COMPSs/JVM boilerplate so the snippet
    focuses on the actual Python traceback or COMPSs error.
    """
    text = _clean(stderr or '')
    # Drop [Timeout] prefix marker
    if text.startswith('[Timeout]'):
        text = text[len('[Timeout]'):].lstrip()
    # Filter boilerplate lines, keep the rest
    meaningful = [
        line for line in text.splitlines()
        if not _BOILERPLATE_LINE.match(line)
    ]
    return '\n'.join(meaningful[-n_lines:])

def _first_exception_line(text: str) -> str:
    """Return the last/most-specific exception line in a Python traceback."""
    hits = re.findall(
        r'^([\w.]*(?:Error|Exception|Warning)[^\n]*)',
        text, re.MULTILINE,
    )
    return hits[-1].strip() if hits else ''

def _first_meaningful_line(text: str) -> str:
    """First non-boilerplate, non-empty line, used for 'other' details."""
    skip = {
        'Picked up JAVA_TOOL_OPTIONS',
        'WARNING: Runtime environment',
        'WARNING: COMPSs Properties',
        'WARNING: Unexpected argument',
        'Error running application',
        'Traceback (most recent call last)',
    }
    for line in text.splitlines():
        line = line.strip()
        if line and not any(line.startswith(s) for s in skip):
            return line[:120]
    return ''

# ── Matcher infrastructure ────────────────────────────────────────────────────

@dataclass
class Matcher:
    category: str
    match:  Callable[[str, dict], bool]   # (cleaned_stderr, run_dict) -> bool
    detail: Callable[[str, dict], str]    # (cleaned_stderr, run_dict) -> str

def classify(stderr: str, run: dict,
             language: str = '', parallelism_model: str = '') -> tuple[str, str]:
    s = _clean(stderr)
    for m in matchers_for(language, parallelism_model):
        try:
            if m.match(s, run):
                return m.category, m.detail(s, run)
        except Exception:
            pass
    return 'other', _first_meaningful_line(s)

# ── Detail extractors ─────────────────────────────────────────────────────────

def _timeout_detail(s, run):
    inner = s[s.index(']') + 1:].strip() if ']' in s[:20] else s
    # Python sub-exception inside the timeout output?
    exc = _first_exception_line(inner)
    if exc:
        return f'timeout + {exc[:80]}'
    # COMPSs task completely failed?
    m = re.search(r"Task '([^']+)' TOTALLY FAILED", inner)
    if m:
        return f'timeout + task_failed: {m.group(1)}'
    # ERRMGR last warning?
    errs = re.findall(r'\[ERRMGR\][^\n]+', inner)
    if errs:
        return f'timeout + {errs[-1].strip()[:80]}'
    return 'timeout'

def _syntax_detail(s, run):
    m = re.search(r'SyntaxError: (.+)', s)
    msg = m.group(1).strip() if m else 'SyntaxError'
    # Distinguish common sub-types
    if 'never closed' in msg:
        bracket = re.search(r"'(.)'", msg)
        return f'unclosed {bracket.group(1) if bracket else "bracket"}'
    if 'unterminated string' in msg:
        return 'unterminated string literal'
    if '__future__' in msg:
        return '__future__ not at top'
    if 'duplicate argument' in msg:
        return 'duplicate argument'
    if 'invalid syntax' in msg:
        # Try to give the surrounding context
        ctx = re.search(r'invalid syntax.*?\((.+?), line \d+\)', msg)
        return f'invalid syntax ({ctx.group(1) if ctx else ""})'
    return msg[:80]

def _import_detail(s, run):
    m = re.search(r"cannot import name '([^']+)' from '([^']+)'", s)
    if m:
        name, module = m.group(1), m.group(2)
        # Real symbol under the wrong module vs a name that exists nowhere.
        kind = 'wrong module path' if name in _PYCOMPSS_API_SYMBOLS else 'hallucinated symbol'
        return f"{kind}: '{name}' from '{module}'"
    m = re.search(r"No module named '([^']+)'", s)
    if m:
        return f"no module '{m.group(1)}'"
    return ''

def _api_namespace_detail(s, run):
    m = re.search(r"module '([\w.]+)' has no attribute '(\w+)'", s)
    if m:
        return f"'{m.group(1)}.{m.group(2)}' (use @task directly)"
    if "'module' object is not callable" in s:
        # Which module?
        m2 = re.search(r'@([\w.]+)\(', s)
        return f"module as callable: @{m2.group(1)}" if m2 else 'module as callable'
    return ''

def _is_future_misuse(s):
    """True when the traceback blames the pycompss Future class or a hallucinated resolver.

    Capital-F 'Future' is the pycompss class (won't match '__future__' or a
    'future_x' variable); '.result()'/'.get()' are the invented resolution calls
    models write instead of compss_wait_on().
    """
    return bool(re.search(r"\bFuture\b", s) or re.search(r"\.result\(\)|\.get\(\)", s))

def _future_detail(s, run):
    for pat, label in [
        (r'TypeError: unsupported operand.*Future', 'arithmetic on future'),
        (r"'Future' object is not iterable",        'iterate future'),
        (r"'Future' object is not subscriptable",   'index future'),
        (r"object has no attribute 'get'",          '.get() on future'),
        (r"object has no attribute 'result'",       '.result() on future'),
        (r"int\(\).*Future|float\(\).*Future",      'type-convert future'),
    ]:
        if re.search(pat, s, re.IGNORECASE):
            return label
    exc = _first_exception_line(s)
    return f'future: {exc[:80]}' if exc else 'unresolved future'

def _entrypoint_detail(s, run):
    return "name 'main' is not defined"

def _missing_api_symbol_detail(s, run):
    m = re.search(r"name '(\w+)' is not defined", s)
    return f"undefined '{m.group(1)}'" if m else 'undefined API symbol'

def _wrong_signature_detail(s, run):
    m = re.search(r"(\w+)\(\) takes (\d+) positional arguments? but (\d+)", s)
    if m:
        return f"{m.group(1)}(): expected {m.group(2)} args, got {m.group(3)}"
    m = re.search(r"takes (\d+) positional arguments? but (\d+)", s)
    if m:
        return f"expected {m.group(1)} args, got {m.group(2)}"
    return 'wrong signature'

def _nonetype_detail(s, run):
    for pat, label in [
        (r"'NoneType' object has no attribute '(\w+)'", lambda m: f'.{m.group(1)} on None'),
        (r"'NoneType' object is not callable",          lambda m: 'None called as function'),
        (r"'NoneType' object is not subscriptable",     lambda m: 'None subscripted'),
        (r"'NoneType' object is not iterable",          lambda m: 'None iterated'),
    ]:
        m = re.search(pat, s)
        if m:
            fn = label
            return fn(m) if callable(fn) else fn
    return "'NoneType' error"

def _pycompss_runtime_detail(s, run):
    m = re.search(r'COMPSs Exception[:\s]+(.+)', s)
    if m:
        return f'COMPSs: {m.group(1).strip()[:80]}'
    # ERRMGR task failure: extract task name if present
    m = re.search(r"Task '([^']+)' TOTALLY FAILED", s)
    if m:
        return f'task totally failed: {m.group(1)}'
    errs = re.findall(r'\[ERRMGR\][^\n]+', s)
    return errs[-1].strip()[:80] if errs else 'COMPSs runtime error'

def _java_detail(s, run):
    m = re.search(r'(java\.[\w.]+(?:Exception|Error)[^\n]*)', s)
    if m:
        return m.group(1).strip()[:80]
    m = re.search(r'Exception in thread "([^"]+)" ([\w.]+)', s)
    if m:
        return f'thread "{m.group(1)}": {m.group(2)}'
    return 'Java exception'

def _unexpected_arg_detail(s, run):
    args = re.findall(r"WARNING: Unexpected argument: (\w+)", s)
    unique = list(dict.fromkeys(args))  # deduplicated, order preserved
    return f'bad @task param: {", ".join(unique[:5])}'

def _type_error_detail(s, run):
    m = re.search(r'TypeError: (.+)', s)
    return m.group(1).strip()[:80] if m else 'TypeError'

def _name_error_detail(s, run):
    m = re.search(r"NameError: name '(\w+)' is not defined", s)
    return f"undefined '{m.group(1)}'" if m else 'NameError'

def _attribute_error_detail(s, run):
    m = re.search(r'AttributeError: (.+)', s)
    return m.group(1).strip()[:80] if m else 'AttributeError'

def _runtime_error_detail(s, run):
    m = re.search(r'RuntimeError: (.+)', s)
    return m.group(1).strip()[:80] if m else 'RuntimeError'

def _value_error_detail(s, run):
    m = re.search(r'ValueError: (.+)', s)
    return m.group(1).strip()[:80] if m else 'ValueError'

# Known PyCOMPSs API symbols that models use without importing correctly.
# Undefined references to these are import/API errors, not entry-point errors.
_PYCOMPSS_API_SYMBOLS = {
    'task', 'constraint', 'mpi', 'binary', 'implement', 'parameter',
    'IN', 'OUT', 'INOUT', 'CONCURRENT',
    'COLLECTION', 'COLLECTION_IN', 'COLLECTION_OUT',
    'FILE_IN', 'FILE_OUT', 'FILE_INOUT',
    'DIRECTORY_IN', 'DIRECTORY_OUT', 'DIRECTORY_INOUT',
    'Type', 'List', 'Dict', 'Tuple',
    'compss_wait_on', 'compss_barrier', 'compss_open',
    'compss_delete_object', 'compss_function',
    'stream_array', 'compss', 'ON_FAILURE', 'returns',
}

# ── MATCHERS registries ───────────────────────────────────────────────────────

# Language-agnostic outcomes (run flags / [Timeout] prefix); shared by every pair.
GENERIC_MATCHERS: list[Matcher] = [
    # ── Outcome-based (must come before content-based) ────────────────────────
    Matcher('success',
            match=lambda s, r: bool(r.get('did_run') and r.get('is_valid')),
            detail=lambda s, r: ''),
    # ran to completion but the result failed the driver's output validation
    Matcher('validation',
            match=lambda s, r: bool(r.get('did_run') and r.get('is_valid') is False),
            detail=lambda s, r: ''),

    # ── Timeout (before content matchers; [Timeout] prefix is unambiguous) ───
    Matcher('timeout',
            match=lambda s, r: s.startswith('[Timeout]') and not _is_future_misuse(s),
            detail=_timeout_detail),
]

# Catch-all, appended last for every pair.
OTHER_MATCHER = Matcher('other',
                        match=lambda s, r: True,
                        detail=lambda s, r: _first_meaningful_line(s))

# Python + PyCOMPSs content matchers, in specificity order (order is load-bearing).
PYCOMPSS_MATCHERS: list[Matcher] = [

    # ── Python / PyCOMPSs import layer ────────────────────────────────────────
    Matcher('import_error',
            match=lambda s, r: bool(
                'cannot import name' in s or 'No module named' in s),
            detail=_import_detail),
    Matcher('api_namespace_misuse',
            match=lambda s, r: bool(
                "'module' object is not callable" in s
                or re.search(r"module '[\w.]+' has no attribute", s)),
            detail=_api_namespace_detail),
    # ── Future / async misuse ─────────────────────────────────────────────────
    # Above unexpected_task_param: a hung future misuse carries COMPSs boilerplate
    # ('Unexpected argument') in its tail that would otherwise steal it.
    Matcher('unresolved_future',
            match=lambda s, r: _is_future_misuse(s),
            detail=_future_detail),

    Matcher('unexpected_task_param',       # @task(in_collection=...) etc.
            match=lambda s, r: 'Unexpected argument' in s and '@task' not in s,
            detail=_unexpected_arg_detail),

    # ── Entry point: main() not found at all ──────────────────────────────────
    Matcher('wrong_entrypoint',
            match=lambda s, r: bool(re.search(r"name 'main' is not defined", s)),
            detail=_entrypoint_detail),

    # ── Entry point: main() exists but with wrong signature ───────────────────
    Matcher('wrong_signature',
            match=lambda s, r: 'positional argument' in s and 'takes' in s,
            detail=_wrong_signature_detail),

    # ── PyCOMPSs API symbol used without being imported ───────────────────────
    Matcher('missing_api_symbol',
            match=lambda s, r: bool(
                re.search(r"name '(\w+)' is not defined", s)
                and any(
                    sym == (m.group(1) if (m := re.search(r"name '(\w+)' is not defined", s)) else '')
                    for sym in _PYCOMPSS_API_SYMBOLS
                )),
            detail=_missing_api_symbol_detail),

    # ── None / uninitialized (a plain Python exception) ───────────────────────
    Matcher('python_exception',
            match=lambda s, r: "'NoneType' object" in s,
            detail=_nonetype_detail),

    # ── Syntax ────────────────────────────────────────────────────────────────
    Matcher('syntax_error',
            match=lambda s, r: 'SyntaxError' in s,
            detail=_syntax_detail),

    # ── PyCOMPSs / COMPSs runtime ─────────────────────────────────────────────
    Matcher('compss_error',
            match=lambda s, r: bool(
                'COMPSs Exception' in s
                or ('ERRMGR' in s and 'TOTALLY FAILED' in s)),
            detail=_pycompss_runtime_detail),

    # ── Java layer (often absorbed by timeout; kept for non-timeout cases) ────
    Matcher('java_exception',
            match=lambda s, r: bool(
                re.search(r'java\.[\w.]+(?:Exception|Error)', s)
                or 'Exception in thread' in s),
            detail=_java_detail),

    # ── General Python exceptions; colon omitted to survive stderr truncation ─
    # Future misuse is caught earlier (unresolved_future), so these are the plain,
    # non-future Python faults, folded under one category.
    Matcher('python_exception',
            match=lambda s, r: 'TypeError' in s,
            detail=_type_error_detail),
    Matcher('python_exception',     # remaining undefined names (not main, not API)
            match=lambda s, r: 'NameError' in s or 'is not defined' in s,
            detail=_name_error_detail),
    Matcher('python_exception',
            match=lambda s, r: 'AttributeError' in s,
            detail=_attribute_error_detail),
    Matcher('python_exception',
            match=lambda s, r: 'ValueError' in s,
            detail=_value_error_detail),
    Matcher('runtime_error',
            match=lambda s, r: 'RuntimeError' in s,
            detail=_runtime_error_detail),

    # ── Generic COMPSs failure with no identifiable Python exception ──────────
    Matcher('compss_error',
            match=lambda s, r: 'Error running application' in s,
            detail=lambda s, r: 'COMPSs terminated with no traceable Python exception'),
]

# Content matchers keyed by (language, parallelism_model). Add new pairs here.
MATCHER_SETS: dict[tuple[str, str], list[Matcher]] = {
    ('python', 'pycompss'): PYCOMPSS_MATCHERS,
    # ('cpp', 'cuda'): CUDA_MATCHERS,   # write these when such results exist
}


def matchers_for(language: str, parallelism_model: str) -> list[Matcher]:
    """Matchers for a (language, parallelism_model) pair: shared + pair-specific + catch-all.

    Pairs absent from MATCHER_SETS fall back to the shared outcomes plus the
    catch-all, so an unsupported language is still split by outcome, not diagnosed.
    """
    specific = MATCHER_SETS.get((language, parallelism_model), [])
    return GENERIC_MATCHERS + specific + [OTHER_MATCHER]

# ── Category grouping ─────────────────────────────────────────────────────────

CATEGORY_GROUP: dict[str, str] = {
    'success':                 'success',
    # a relaxation actually rescued this output (grounded in relaxations_applied)
    'recoverable':             'recoverable',
    # ran but produced an incorrect result (own coarse outcome)
    'validation':              'validation',
    # runtime: any failed run; unfixed import/api faults land here, not recoverable.
    'import_error':            'runtime',
    'api_namespace_misuse':    'runtime',
    'unexpected_task_param':   'runtime',
    'missing_api_symbol':      'runtime',
    'wrong_entrypoint':        'runtime',
    'unresolved_future':       'runtime',
    'syntax_error':            'runtime',
    'wrong_signature':         'runtime',
    'python_exception':        'runtime',
    'runtime_error':           'runtime',
    'timeout':                 'runtime',
    'compss_error':            'runtime',
    'java_exception':          'runtime',
    # rejected before running (no @task decorator); see category_group() docstring
    'build_failure':           'build',
    'no_runs':                 'other',
    'not_evaluated':           'other',
    'other':                   'other',
}

def category_group(category: str) -> str:
    """Coarse outcome group for a fine category.

    - success:     ran and passed validation natively (no relaxation applied).
    - recoverable: failed natively but a relaxation in drivers/python/relaxations.py
                   actually rescued it (the output carries a non-empty
                   relaxations_applied). The reported pass@k does not count these as
                   passes. This is the demonstrated fix set, not a guess: an unfixed
                   import/api fault stays under 'runtime', not here.
    - validation:  ran to completion but the result was wrong (failed validation).
    - build:       for pycompss, the generated code had no @task decorator, so the
                   harness rejected it before assembling/running it (python is not
                   compiled; the "build" step is a file merge gated by that check).
    - runtime: any other failed run (crash, syntax fault, or an unfixed
                   import/namespace/future error no relaxation cleared here).
    - other:       unevaluated, or a failure with no traceable error text.
    """
    return CATEGORY_GROUP.get(category, 'other')

# ── I/O helpers ──────────────────────────────────────────────────────────────

def resolve_dir(value: str) -> str:
    if os.path.isdir(value):
        return value
    candidate = os.path.join(
        os.path.dirname(__file__), '..', 'drivers', 'outputs', value)
    candidate = os.path.normpath(candidate)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"'{value}' is not a valid path and was not found under drivers/outputs/")

def run_name(driver_outputs_dir: str) -> str:
    return os.path.basename(os.path.normpath(driver_outputs_dir))

def output_path(driver_outputs_dir: str) -> str:
    name = run_name(driver_outputs_dir)
    return os.path.normpath(os.path.join(
        os.path.dirname(__file__), 'outputs', name, f'errors_{name}.csv'))

# ── Core processing ───────────────────────────────────────────────────────────

def process_file(json_path: str, include_success: bool, snippet_lines: int = 20) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)

    model = (os.path.basename(json_path)
             .replace('output_drivers_', '').replace('.json', ''))
    rows = []

    for entry in data:
        if not isinstance(entry.get('outputs'), list):
            continue
        language = entry.get('language', '')
        parallelism_model = entry.get('parallelism_model', '')
        meta = {
            'model':             model,
            'problem_name':      entry.get('name', ''),
            'problem_type':      entry.get('problem_type', ''),
            'language':          language,
            'parallelism_model': parallelism_model,
        }

        for oi, out in enumerate(entry['outputs']):
            if not isinstance(out, dict):
                # Bare string, not yet evaluated
                rows.append({**meta, 'output_idx': oi, 'run_idx': None,
                             'did_build': None, 'did_run': None, 'is_valid': None,
                             'error_category': 'not_evaluated', 'error_detail': '',
                             'stderr_snippet': ''})
                continue

            did_build = out.get('did_build', False)

            if not did_build:
                build_stderr = out.get('build_stderr', '') or ''
                rows.append({**meta, 'output_idx': oi, 'run_idx': None,
                             'did_build': False, 'did_run': False, 'is_valid': None,
                             'error_category': 'build_failure',
                             'error_detail': build_stderr[:120].replace('\n', ' '),
                             'stderr_snippet': _stderr_snippet(build_stderr, snippet_lines)})
                continue

            runs = out.get('runs') or []
            if not runs:
                rows.append({**meta, 'output_idx': oi, 'run_idx': None,
                             'did_build': True, 'did_run': False, 'is_valid': None,
                             'error_category': 'no_runs', 'error_detail': '',
                             'stderr_snippet': ''})
                continue

            relaxations = out.get('relaxations_applied') or []
            for ri, run in enumerate(runs):
                stderr = run.get('stderr', '') or ''
                category, detail = classify(stderr, run, language, parallelism_model)
                # Relaxation-rescued pass: not a native pass, attributed to the fix.
                if category == 'success' and relaxations:
                    category = 'recoverable'
                    detail = ', '.join(relaxations)
                if category in ('success', 'recoverable') and not include_success:
                    continue
                rows.append({
                    **meta,
                    'output_idx':     oi,
                    'run_idx':        ri,
                    'did_build':      did_build,
                    'did_run':        run.get('did_run'),
                    'is_valid':       run.get('is_valid'),
                    'error_category': category,
                    'error_detail':   detail,
                    'stderr_snippet': _stderr_snippet(stderr, snippet_lines),
                })

    return rows

# ── Summarization ────────────────────────────────────────────────────────────

def _topk_pivot(df: pd.DataFrame, group_col: str, k: int) -> pd.DataFrame:
    """For each value of group_col, return the top-k error categories by count,
    with count and percentage of that group's total runs."""
    # success and recoverable are passing outcomes, not failure modes.
    total = df.groupby(group_col).size().rename('total_runs')
    errors = df[~df['error_category'].isin(['success', 'recoverable'])]
    counts = (errors.groupby([group_col, 'error_category'])
                    .size()
                    .reset_index(name='count'))
    counts = counts.merge(total.reset_index(), on=group_col)
    counts['pct_of_runs'] = (counts['count'] / counts['total_runs'] * 100).round(1)
    counts['rank'] = (counts.groupby(group_col)['count']
                            .rank(method='first', ascending=False)
                            .astype(int))
    return (counts[counts['rank'] <= k]
            .sort_values([group_col, 'rank'])
            .drop(columns='rank')
            .reset_index(drop=True))


def _category_pivot(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Wide pivot: rows = group_col values, columns = error categories, values = counts."""
    counts = (df.groupby([group_col, 'error_category'])
                .size()
                .unstack(fill_value=0))
    # Add total and sort by total errors (success and recoverable are not failures)
    error_cols = [c for c in counts.columns if c not in ('success', 'recoverable')]
    counts['total_errors'] = counts[error_cols].sum(axis=1)
    counts['total_runs']   = counts.sum(axis=1) - counts.get('total_errors', 0)
    return counts.sort_values('total_errors', ascending=False).reset_index()


def _group_composition(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per group_col value: run count and share of runs in each category_group."""
    total = df.groupby(group_col).size().rename('total_runs')
    comp = (df.groupby([group_col, 'category_group']).size()
              .unstack(fill_value=0))
    comp = comp.div(comp.sum(axis=1), axis=0)          # row-normalised shares
    comp = comp.merge(total, left_index=True, right_index=True)
    return comp.sort_values('recoverable', ascending=False).reset_index() \
        if 'recoverable' in comp.columns else comp.reset_index()


def _digest_md(df: pd.DataFrame, run_name: str, top_k: int = 3) -> str:
    """Markdown summary: per-model outcome split plus dominant failure modes."""
    lines = [f'# Failure-mode summary: {run_name}', '']
    comp = _group_composition(df, 'model')
    top = _topk_pivot(df, 'model', top_k)

    def pct(row, col):
        return f'{100 * row[col]:.0f}%' if col in row and pd.notna(row[col]) else '0%'

    lines += ['## Per-model outcome composition (share of run attempts)', '',
              '| model | runs | success | recoverable | runtime | build | top failure modes |',
              '|---|---:|---:|---:|---:|---:|---|']
    for _, row in comp.iterrows():
        model = row['model']
        modes = top[top['model'] == model].head(top_k)
        modes_str = '; '.join(
            f"{m['error_category']} ({m['pct_of_runs']:.0f}%)" for _, m in modes.iterrows())
        lines.append(
            f"| {model} | {int(row['total_runs'])} | {pct(row, 'success')} | "
            f"{pct(row, 'recoverable')} | {pct(row, 'runtime')} | "
            f"{pct(row, 'build')} | {modes_str} |")

    lines += ['', '## Dominant failure mode per problem type', '',
              '| problem type | top failure modes |', '|---|---|']
    tp = _topk_pivot(df, 'problem_type', top_k)
    for ptype, grp in tp.groupby('problem_type'):
        modes_str = '; '.join(
            f"{m['error_category']} ({m['pct_of_runs']:.0f}%)"
            for _, m in grp.head(top_k).iterrows())
        lines.append(f'| {ptype} | {modes_str} |')
    lines.append('')
    return '\n'.join(lines)


def summarize(errors_csv: str, out_dir: str, top_k: int, write_digest: bool = False) -> None:
    df = pd.read_csv(errors_csv)
    if 'category_group' not in df.columns:
        df['category_group'] = df['error_category'].map(category_group)

    tables = {
        'topk_by_model':        (_topk_pivot(df, 'model', top_k),
                                 f'Top-{top_k} error categories per model'),
        'topk_by_problem_type': (_topk_pivot(df, 'problem_type', top_k),
                                 f'Top-{top_k} error categories per problem type'),
        'topk_by_problem':      (_topk_pivot(df, 'problem_name', top_k),
                                 f'Top-{top_k} error categories per problem'),
        'pivot_model':          (_category_pivot(df, 'model'),
                                 'Error category counts by model × category'),
        'pivot_problem_type':   (_category_pivot(df, 'problem_type'),
                                 'Error category counts by problem_type × category'),
        'group_by_model':       (_group_composition(df, 'model'),
                                 'Category-group shares by model'),
        'group_by_problem_type':(_group_composition(df, 'problem_type'),
                                 'Category-group shares by problem_type'),
    }

    os.makedirs(out_dir, exist_ok=True)
    for key, (table, desc) in tables.items():
        path = os.path.join(out_dir, f'errors_{key}.csv')
        table.to_csv(path, index=False)
        print(f'  [{desc}] → {path}')

    if write_digest:
        digest_path = os.path.join(out_dir, f'errors_{run_name(out_dir)}_digest.md')
        with open(digest_path, 'w') as f:
            f.write(_digest_md(df, run_name(out_dir), top_k))
        print(f'  [Markdown digest] → {digest_path}')

    # Print the two most useful tables to stdout for a quick overview
    print('\n── Top errors by model ──')
    print(_topk_pivot(df, 'model', min(top_k, 3)).to_string(index=False))
    print('\n── Top errors by problem type ──')
    print(_topk_pivot(df, 'problem_type', min(top_k, 3)).to_string(index=False))


# ── Main ─────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dir',
        help="Run name (e.g. 'kernel-20') or full path to drivers/outputs/<dir>/.")
    parser.add_argument('-o', '--output', default=None,
        help='Output CSV path. Defaults to analysis/outputs/<dir>/errors_<dir>.csv.')
    parser.add_argument('--include-success', action='store_true',
        help='Include successful runs in the output (excluded by default).')
    parser.add_argument('--snippet-lines', type=int, default=20, metavar='N',
        help='Number of trailing meaningful stderr lines to keep in stderr_snippet (default: 20).')
    parser.add_argument('--summarize', action='store_true',
        help='After classification, also write summary pivot tables.')
    parser.add_argument('--only-summarize', action='store_true',
        help='Skip classification; read existing errors CSV and write summaries only.')
    parser.add_argument('--digest', action='store_true',
        help='With --summarize, also write a Markdown digest (errors_<dir>_digest.md).')
    parser.add_argument('--top-k', type=int, default=5, metavar='K',
        help='Number of top error categories to show in per-model/type summaries (default: 5).')
    return parser.parse_args()


def main():
    args = get_args()
    driver_dir = resolve_dir(args.dir)
    out_path   = args.output or output_path(driver_dir)
    out_dir    = os.path.dirname(out_path)

    if not args.only_summarize:
        files = sorted(
            os.path.join(driver_dir, f)
            for f in os.listdir(driver_dir)
            if f.startswith('output_drivers_') and f.endswith('.json')
        )
        if not files:
            raise FileNotFoundError(f'No output_drivers_*.json files in {driver_dir}')

        all_rows = []
        for fpath in files:
            rows = process_file(fpath, args.include_success, args.snippet_lines)
            all_rows.extend(rows)
            model = os.path.basename(fpath).replace('output_drivers_', '').replace('.json', '')
            print(f'  {model}: {len(rows)} rows')

        df = pd.DataFrame(all_rows)
        df['category_group'] = df['error_category'].map(category_group)
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f'\nSaved {len(df)} rows → {out_path}')

        print('\nError category breakdown:')
        print(df.groupby('error_category').size().sort_values(ascending=False).to_string())

    if args.summarize or args.only_summarize:
        if not os.path.exists(out_path):
            raise FileNotFoundError(
                f'Errors CSV not found: {out_path}\n'
                f'Run without --only-summarize first to generate it.')
        print(f'\nSummarizing {out_path} ...')
        summarize(out_path, out_dir, args.top_k, write_digest=args.digest)


if __name__ == '__main__':
    main()
