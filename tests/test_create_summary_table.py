import importlib.util
import io
import tarfile
from pathlib import Path

import pytest


def load_create_summary_table_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "create_summary_table.py"
    spec = importlib.util.spec_from_file_location("create_summary_table", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


create_summary_table = load_create_summary_table_module()


def write_tar_member(archive_path, member_name, payload=b"contents"):
    with tarfile.open(archive_path, "w:gz") as tar:
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))


def test_extract_results_rejects_members_outside_extraction_directory(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    archive_path = sandbox / "results.tar.gz"
    write_tar_member(archive_path, "../escaped.txt")

    with pytest.raises(ValueError, match="Unsafe path"):
        create_summary_table.extract_results(str(archive_path))

    assert not (tmp_path / "escaped.txt").exists()


def test_extract_results_extracts_safe_archive_members(tmp_path):
    archive_path = tmp_path / "results.tar.gz"
    write_tar_member(archive_path, "results/dataset/train-8-0/results.json", b'{"score": 95}')

    extracted_path = create_summary_table.extract_results(str(archive_path))

    assert extracted_path == str(tmp_path / "results")
    assert (tmp_path / "results" / "dataset" / "train-8-0" / "results.json").read_bytes() == b'{"score": 95}'
