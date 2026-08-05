"""CSV 입출력 + 스키마 검증 + 실행시간 측정 + 입력 provenance.

핵심 규칙:
- 쓰기 시 컬럼 목록을 schema.py 상수와 대조하여, 누락/오타 컬럼이 있으면 즉시 실패.
  (병합 단계에서 발견되는 것보다 각 학생의 실행 단계에서 실패하는 것이 싸다)
- append 는 지원하지 않는다. 각 실행은 전체 파일을 원자적으로 교체한다.
- 실행 기록에는 입력 파일의 내용 해시를 남긴다 (input_provenance 참조).
"""
import csv
import hashlib
import os
import time
from contextlib import contextmanager
from typing import Iterable


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
    # utf-8-sig: prompts.csv 와 동일 인코딩으로 통일 (BOM 없으면 엑셀에서 한글 깨짐).
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=expected_columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: str, expected_columns: list[str] | None = None) -> list[dict]:
    # utf-8-sig: BOM 이 있으면 벗기고, 없어도(plain utf-8) 그대로 읽는다.
    with open(path, newline="", encoding="utf-8-sig") as f:
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


def file_sha256(path: str) -> str:
    """파일 내용의 sha256. 큰 파일도 메모리에 통째로 올리지 않는다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def input_provenance(paths: Iterable[str]) -> dict:
    """실행 메타데이터에 넣을 입력 파일 기록: 경로 · 내용 해시 · 크기.

    왜 필요한가:
      run_metadata 에 git_commit 만 남기면 "어떤 내용으로 돌렸는가"가 증명되지 않는다.
      prompts.csv 는 작업 트리 파일이라 커밋되지 않은 수정이 섞일 수 있고, 결과 파일을
      부분 재실행(--resume)하면 **한 파일 안에 서로 다른 입력 버전의 행이 섞인다.**
      해시가 있으면 그 상황을 사후에 발견할 수 있다.

      실제로 필요해진 사례: UNSAFE_CRIM_24 의 표현쌍 수정으로 432행 중 18행이 바뀌면서,
      각 팀원이 결과를 부분 재실행하게 되었다. 해시가 없으면 어느 결과가 어느 버전
      기준인지 구분할 방법이 없다.

    사용법 (run_metadata 조립부에서):
        meta["inputs"] = input_provenance([str(PROMPTS_CSV)])

    Returns:
        {상대경로 또는 파일명: {"sha256": ..., "bytes": ...}}
        읽을 수 없는 파일은 {"error": ...} 로 남긴다 — 기록을 남기는 것이 목적이라
        여기서 실행을 중단시키지 않는다.
    """
    out: dict[str, dict] = {}
    for p in paths:
        key = os.path.basename(p)
        try:
            out[key] = {
                "path": p.replace("\\", "/"),
                "sha256": file_sha256(p),
                "bytes": os.path.getsize(p),
            }
        except OSError as e:
            out[key] = {"path": p.replace("\\", "/"), "error": str(e)}
    return out


@contextmanager
def timed():
    """runtime_ms 측정용.  usage:  with timed() as t: ...;  t['ms']"""
    box = {}
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = round((time.perf_counter() - t0) * 1000, 3)
