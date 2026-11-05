# Alpha research scripts

Research utilities in `alphas2/` and `alphas_zero/`. Real provider calls require
your own service accounts and dependencies; CI never contacts those services.

## Local checks

With Python 3.11–3.13, run the same offline checks as GitHub Actions:

```sh
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

The checks parse all tracked Python sources, reject known hardcoded provider-key
patterns and credential files, and test provider dispatch with mocked requests.
They do not validate alpha profitability, paid APIs, or live submissions.

## Credentials

Restore WorldQuant login files locally from your private credential backup when
needed. They are ignored by Git. Provider methods read `DOUBAO_API_KEY`,
`DASHSCOPE_API_KEY`, `TENCENTCLOUD_SECRET_ID`, and `TENCENTCLOUD_SECRET_KEY` from
the process environment. Never commit their values. Previously published
credentials should be rotated with their providers; removing Git history does
not revoke an exposed credential.

`alphas2.call_api.ProviderAPIMixin` contains reusable provider calls. Importing it
does not submit requests. The generator scripts require their documented Python
packages and live service access for actual research.

## Deployment

No automatic deployment is configured. GitHub CI validates offline code only;
running research or submitting alphas remains an explicit local operation.
