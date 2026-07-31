"""CSV 입출력 + 스키마 검증 + 실행시간 측정.

핵심 규칙:
- 쓰기 시 컬럼 목록을 schema.py 상수와 대조하여, 누락/오타 컬럼이 있으면 즉시 실패.
  (병합 단계에서 발견되는 것보다 각 학생의 실행 단계에서 실패하는 것이 싸다)
- append 는 지원하지 않는다. 각 실행은 전체 파일을 원자적으로 교체한다.
"""
import csv
import os
import time
from contextlib import contextmanager


class SchemaMismatchError(Exception):
    pass


def _check_columns(rows: list[dict], expected_columns: list[str], where: str) -> None:
    expected = set(expected_columns)
    for i, row in enumerate(rows):
        got = set(row.keys())
        missing = expected - got
        extra = got - expected
        if missing or extra:
            raise SchemaMismatchError(
                f"{where}: row {i} column mismatch. "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )


def write_csv(path: str, rows: list[dict], expected_columns: list[str]) -> None:
    """schema 검증 후 원자적으로 쓴다 (tmp → rename)."""
    _check_columns(rows, expected_columns, where=path)
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=expected_columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: str, expected_columns: list[str] | None = None) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if expected_columns is not None:
        _check_columns(rows, expected_columns, where=path)
    return rows


def check_primary_key(rows: list[dict], key_columns: list[str], where: str) -> None:
    """PK 유일성 검증. 병합 전 각 산출물에서 반드시 호출."""
    seen: set[tuple] = set()
    for i, row in enumerate(rows):
        key = tuple(row[c] for c in key_columns)
        if key in seen:
            raise SchemaMismatchError(f"{where}: duplicate PK {key} at row {i}")
        seen.add(key)


@contextmanager
def timed():
    """runtime_ms 측정용.  usage:  with timed() as t: ...;  t['ms']"""
    box = {}
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = round((time.perf_counter() - t0) * 1000, 3)
