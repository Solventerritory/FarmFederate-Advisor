# Codebase cleanup — 2026-09-06

Removed 284 files (37,625,608 bytes) after checking references and duplicate
contents. Six small photo assets were added to keep both Overleaf packages
self-contained, for about 36.9 MB net reduction before documentation changes.
This reduces working-tree content; Git history has not been rewritten.

## Removed content

| Category | Files | Reason |
| --- | ---: | --- |
| Nested `data_final/data_final/` bundle | 208 | Every file matched its canonical counterpart in `data_final/` byte-for-byte |
| Manuscript migration/layout scripts | 36 | One-time patches; no surviving script, notebook, or workflow refers to them |
| Manuscript scratch extracts | 8 | Stale copies of text already represented by manuscript sources/PDFs |
| PDF preview images | 12 | Reproducible render output |
| Unused duplicate Overleaf figures | 13 | Not referenced by any retained TeX source; identical copies remain elsewhere |
| Nested figure-only dataset in slim package | 3 | Relocated the required photographs to `plots/photos/` |
| Duplicate dataset-building script | 1 | Root `build_dataset_remaining.py` is identical and retained |
| Explicitly deprecated hardware helpers | 3 | No retained imports, workflows, or scripts reference them |

## References and documentation

- Updated the cross-domain experiment's field-image default to
  `data_final/real_dataset_sorted`.
- Updated both sets of architecture diagrams to load their three photographs
  from `plots/photos/`. This also fixes a pre-existing missing-photo compilation
  failure in the full package.
- Replaced the obsolete results-archive README with a map of the active code,
  datasets, experiment outputs, and paper build instructions.
- Corrected the Overleaf README and ignored disposable test caches and PDF
  inspection scratch files.

The current `main.tex` files, training implementations, frontend, API, result
records, and canonical dataset contents were preserved. Distinct application
submission bundles, older deliverables, notebook workflows, and experimental
datasets were retained: similar names alone do not establish redundancy.

## Validation

- Verified the recovery archive's SHA-256 hashes before removing any file.
- Verified every consolidated dataset file against its archived original.
- Parsed 178 remaining Python files successfully; no new syntax failures.
- Found no retained Python imports or script/notebook/workflow references to
  removed Python files.
- Built both manuscripts with pdfLaTeX in three passes: eight pages each, with
  no unresolved references. Both `main.tex` files remained byte-identical to
  their pre-cleanup versions.
- Compared every rebuilt PDF page with its existing PDF: all 16 pages were
  pixel-identical, and extracted text matched. Visually inspected the repaired
  full-package architecture page.
- Started `backend.demo_server` through FastAPI TestClient; `/health` and
  `/models` both returned HTTP 200.
- `git diff --check` passed.

Training was not rerun, and the full dependency-heavy test suite was not run.
Three legacy scripts already had syntax errors in Git HEAD and remain unchanged:
`backend/FarmFederate_RealData_Colab.py`, `backend/farm_advisor_complete.py`, and
`backend/federated_complete_system.py`.

## Recovery

The local recovery folder is outside the repository:

```text
/Users/ayushdebnath/FarmFederate-cleanup-backup-20260906-092815/
```

It contains `manifest.json` with every removed path, reason, size, and SHA-256,
and `removed-and-original-files.tar.gz` with the removed files and originals of
edited documentation/diagram files. The original current manuscripts and PDFs
are also included. The cleanup did not rewrite Git history. This local recovery
archive is not included in the repository.

To recover files, extract the archive into a separate directory first and copy
back the desired paths; extracting directly over the repository could replace
subsequent edits.
