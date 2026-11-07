# 0.0.3 local release review — November 7, 2026

No prior project version was declared; establish 0.0.3. This owned research-script
repository is separate from brain_work and is not a deployment dependency.

Design: correct Hunyuan model attribution. The provider chose and reported a model
randomly, but always sent hunyuan-t1-latest. Use the selected model in the request.
Self-review: preserve retries, credentials and response handling; exercise the
alternative model with a mocked SDK, without installing or contacting that SDK.

Validation: `python scripts/check_repository.py` and
`python -m unittest discover -s tests -v`. Final results are recorded in the
brain_work release report. The SDK test verifies the request, not actual provider
availability. No model recommendation or live compatibility claim is made.

Limitations: research profitability, live provider access and submissions remain
unverified. No deployment, push or running-process change is part of this release.
