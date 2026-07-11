"""Tests dirigidos al endurecimiento de scripts/provision_llama_gpu.py (B47 / C14 a-f).

Cubren lo verificable SIN red ni GPU:
  (a) `_safe_extractall` bloquea tar-slip (path-traversal, rutas absolutas, symlinks
      que escapan del destino) y sigue extrayendo tarballs benignos.
  (b) SHA mismatch -> `cmd_download` retorna != 0 y borra el artefacto corrupto.
  (b/c) `_verify_or_fail`: seteado -> enforce (borra + lanza); vacío -> advertencia
      honesta UNVERIFIED sin fallar.
  (d/e/f) `cmd_write_env`: preserva el contrato de nombres CUDA, resuelve el
      GGML_VK_VISIBLE_DEVICES override, y deja el gate de gobernanza del razonador
      externo como opt-in (comentado por defecto; activo con --enable-external-reasoner).

Re-obra (must-fix de la auditoría adversaria):
  1) gobernanza opt-in por defecto (no auto-escalar el gate en el .env).
  2) integridad del BINARIO llama.cpp (no solo del GGUF) vía `_verify_or_fail`.
  3) `cmd_download` verifica SIEMPRE (fuera del guard exists()): un artefacto
      preexistente corrupto aborta y no reporta "Descarga OK".
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from scripts import provision_llama_gpu as prov


# --------------------------------------------------------------------------- #
# (a) Extracción segura contra tar-slip.
# --------------------------------------------------------------------------- #


def _tar_with_member(tar_path: Path, name: str, payload: bytes = b"pwned") -> None:
    """Crea un tar.gz con un único miembro-archivo de nombre arbitrario `name`."""
    with tarfile.open(tar_path, "w:gz") as tf:
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        import io

        tf.addfile(info, io.BytesIO(payload))


def test_safe_extractall_rejects_parent_traversal(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    tar_path = tmp_path / "evil.tar.gz"
    _tar_with_member(tar_path, "../escape.txt")

    with tarfile.open(tar_path, "r:gz") as tf:
        with pytest.raises(ValueError, match="tar inseguro"):
            prov._safe_extractall(tf, dest)

    # No debe haber escrito NADA fuera de dest.
    assert not (tmp_path / "escape.txt").exists()
    assert list(dest.iterdir()) == []


def test_safe_extractall_rejects_absolute_path(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    tar_path = tmp_path / "abs.tar.gz"
    # Ruta absoluta fuera del destino.
    _tar_with_member(tar_path, str(tmp_path / "outside" / "abs.txt"))

    with tarfile.open(tar_path, "r:gz") as tf:
        with pytest.raises(ValueError, match="tar inseguro"):
            prov._safe_extractall(tf, dest)

    assert not (tmp_path / "outside").exists()


def test_safe_extractall_rejects_escaping_symlink(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    tar_path = tmp_path / "link.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        info = tarfile.TarInfo(name="link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../../../etc/passwd"
        tf.addfile(info)

    with tarfile.open(tar_path, "r:gz") as tf:
        with pytest.raises(ValueError, match="link fuera del destino"):
            prov._safe_extractall(tf, dest)


def test_safe_extractall_allows_benign_tar(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    tar_path = tmp_path / "ok.tar.gz"
    _tar_with_member(tar_path, "sub/ok.txt", payload=b"hello")

    with tarfile.open(tar_path, "r:gz") as tf:
        prov._safe_extractall(tf, dest)

    extracted = dest / "sub" / "ok.txt"
    assert extracted.exists()
    assert extracted.read_bytes() == b"hello"


# --------------------------------------------------------------------------- #
# (b/c) Verificación de integridad fail-closed.
# --------------------------------------------------------------------------- #


def test_verify_or_fail_mismatch_deletes_and_raises(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.gguf"
    artifact.write_bytes(b"real-content")
    real = prov._sha256(artifact)
    wrong = "0" * 64
    assert real != wrong

    with pytest.raises(prov.IntegrityError, match="fail-closed"):
        prov._verify_or_fail(artifact, wrong, "reasoner_gguf")

    # Fail-closed: el artefacto corrupto se borra del disco.
    assert not artifact.exists()


def test_verify_or_fail_match_keeps_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.gguf"
    artifact.write_bytes(b"real-content")
    real = prov._sha256(artifact)

    prov._verify_or_fail(artifact, real, "reasoner_gguf")  # no debe lanzar
    assert artifact.exists()


def test_verify_or_fail_empty_hash_warns_but_does_not_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = tmp_path / "embed.gguf"
    artifact.write_bytes(b"embed-content")

    prov._verify_or_fail(artifact, "", "embed_gguf")  # no debe lanzar
    assert artifact.exists()

    out = capsys.readouterr().out
    assert "UNVERIFIED" in out
    assert "embed_gguf" in out
    # Imprime el hash observado para poder pinnearlo.
    assert prov._sha256(artifact) in out


def test_cmd_download_fail_closed_on_reasoner_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_root = tmp_path / "models"

    def fake_hf_download(repo: str, filename: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"tampered-gguf")
        return dest

    monkeypatch.setattr(prov, "_hf_download", fake_hf_download)
    # Digest observado incorrecto -> mismatch contra REASONER_SHA256 pinneado.
    monkeypatch.setattr(prov, "_sha256", lambda path: "deadbeef" * 8)

    reasoner_path = prov._paths(models_root)["reasoner_gguf"]

    rc = prov.cmd_download(models_root, embeddings=False)

    assert rc != 0
    # El GGUF manipulado NO queda en disco.
    assert not reasoner_path.exists()


# --------------------------------------------------------------------------- #
# (d/e/f) write_env: contrato de nombres, override de device, gobernanza.
# --------------------------------------------------------------------------- #


def _write_env_lines(tmp_path: Path, **kwargs) -> list[str]:
    out = tmp_path / ".env.local"
    rc = prov.cmd_write_env(tmp_path / "models", out, **kwargs)
    assert rc == 0
    return out.read_text(encoding="utf-8").splitlines()


def test_write_env_vk_device_override(tmp_path: Path) -> None:
    lines = _write_env_lines(tmp_path, vk_device="3")
    assert "export GGML_VK_VISIBLE_DEVICES=3" in lines
    assert "export GGML_VK_VISIBLE_DEVICES=1" not in lines


def test_write_env_vk_device_default_is_1(tmp_path: Path) -> None:
    lines = _write_env_lines(tmp_path)
    assert "export GGML_VK_VISIBLE_DEVICES=1" in lines


def test_write_env_external_reasoner_opt_in_commented_by_default(tmp_path: Path) -> None:
    # (must-fix 1) El gate de gobernanza NO se auto-habilita: por defecto las dos
    # claves salen COMENTADAS (opt-in), nunca activas.
    lines = _write_env_lines(tmp_path)
    assert "export RNFE_ALLOW_EXTERNAL_REASONER=1" not in lines
    assert "export RNFE_MAX_COMPUTE_TIER=tier_3_external" not in lines
    text = "\n".join(lines)
    assert "# export RNFE_ALLOW_EXTERNAL_REASONER=1" in text
    assert "# export RNFE_MAX_COMPUTE_TIER=tier_3_external" in text
    # Nota explicativa de una línea para el operador.
    assert "descomentar para habilitar el razonador externo tier_3" in text


def test_write_env_external_reasoner_active_with_flag(tmp_path: Path) -> None:
    # (must-fix 1) Con --enable-external-reasoner las claves se emiten activas.
    lines = _write_env_lines(tmp_path, enable_external_reasoner=True)
    assert "export RNFE_ALLOW_EXTERNAL_REASONER=1" in lines
    assert "export RNFE_MAX_COMPUTE_TIER=tier_3_external" in lines
    text = "\n".join(lines)
    assert "# export RNFE_ALLOW_EXTERNAL_REASONER=1" not in text


def test_main_write_env_external_reasoner_flag_activates(
    tmp_path: Path,
) -> None:
    # (must-fix 1) El flag del CLI llega a cmd_write_env y activa las claves.
    out = tmp_path / ".env.enabled"
    rc = prov.main(
        [
            "--write-env",
            str(out),
            "--models-root",
            str(tmp_path / "models"),
            "--enable-external-reasoner",
        ]
    )
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert "export RNFE_ALLOW_EXTERNAL_REASONER=1" in lines
    assert "export RNFE_MAX_COMPUTE_TIER=tier_3_external" in lines


def test_main_write_env_external_reasoner_default_commented(tmp_path: Path) -> None:
    # (must-fix 1) Sin el flag, el CLI deja las claves comentadas.
    out = tmp_path / ".env.default"
    rc = prov.main(["--write-env", str(out), "--models-root", str(tmp_path / "models")])
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert "export RNFE_ALLOW_EXTERNAL_REASONER=1" not in lines
    assert "export RNFE_MAX_COMPUTE_TIER=tier_3_external" not in lines


def test_write_env_preserves_cuda_contract(tmp_path: Path) -> None:
    # (d) NO renombrar: el config lee estas claves con nombre CUDA.
    text = "\n".join(_write_env_lines(tmp_path))
    assert "export RNFE_EXTERNAL_REASONER_BACKEND=cuda" in text
    assert "RNFE_LLAMA_CLI_CUDA=" in text
    # La deuda "vulkan-bajo-nombre-CUDA" queda inequívoca en el .env generado.
    assert "backend real: vulkan" in text


def test_main_write_env_uses_env_vk_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # (e) El override por env GGML_VK_VISIBLE_DEVICES se propaga al .env via el default del CLI.
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "7")
    out = tmp_path / ".env.fromenv"
    rc = prov.main(["--write-env", str(out), "--models-root", str(tmp_path / "models")])
    assert rc == 0
    assert "export GGML_VK_VISIBLE_DEVICES=7" in out.read_text(encoding="utf-8").splitlines()
