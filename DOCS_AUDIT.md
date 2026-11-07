# Documentation audit — November 7, 2026

Version remains 0.0.3; documentation changes do not imply a new trained model or deployed service. README and DESIGN.md are current guides; RELEASE_REVIEW.md preserves dated 0.0.3 evidence unchanged. The docs checker validates owned Markdown inline local links/heading fragments without network or snippet execution. It excludes third-party/env/cache docs and does not check external URL availability, HTML/reference links or notebook Markdown automatically. A synthetic unittest covers missing target/anchor failures.

## Inventory and source review

All five owned Markdown documents are covered: root README, DESIGN, dated release review, this audit and alphas2/README.md. The nested strategy README is retained upstream/reference content with a current-guide supersession note; its profitability/concurrency claims are not restated as verified implementation. Reviewed alphas_zero script entry points show top-level authentication/file access, so current docs explicitly prohibit treating arbitrary script import/help as an offline healthcheck.

README documents the actual provider method names/default retry count, return tuple, four environment credential names and supported offline Python matrix. DESIGN distinguishes request identity from remote model availability and notes ambiguous/billable retries, script-specific dependencies, log sensitivity and lack of protected submission recovery. No common daemon/API/data delivery contract or brain_work ownership guarantees are invented.

## Verification and limitations

Run `python scripts/check_docs.py`, `python scripts/check_repository.py` and `python -m unittest discover -s tests -v`. Final counts/results are in the publication manifest. Existing CI runs these on Python3.11/3.12/3.13. Tests mock requests/SDK modules; no provider packages/credentials/actual requests, miner imports or native submissions are required. The tracked-file credential check uses limited patterns and cannot prove arbitrary secrets absent or prior credentials revoked. No paid API, model-list availability, strategy profitability or live runtime was verified.
