# Alpha research scripts

Version 0.0.3 contains WorldQuant research scripts in `alphas2/` and `alphas_zero/`, plus offline validation. This is independent of brain_work: it does not share brain_work's supervised writer, database, quota or protected-request ownership guarantees.

## Safe offline checks

Python 3.11–3.13 are the hosted CI versions. From the repository root:

```sh
python3 scripts/check_docs.py
python3 scripts/check_repository.py
python3 -m unittest discover -s tests -v
```

These commands use the standard library; provider tests replace requests/the SDK with mocks. No credential, paid API or BRAIN account is required. The source checker parses tracked Python files and detects known credential patterns/generated files. It is not an exhaustive secret scanner or proof that existing exposed keys are revoked. Tests do not import the executable mining scripts.

## Reusable provider interface

`alphas2/call_api.py` exposes `ProviderAPIMixin`: `call_doubao_api`, `call_hunyuan_api`, `call_qianwen_api`, and `call_random_api`. Each accepts a prompt and `max_retries` (default 3), returning `(model_name, content_or_none)`. Real use requires requests; Hunyuan also imports its Tencent SDK. Model lists are historical source defaults, not verified current availability.

| Provider | Environment variables |
|---|---|
| Doubao | `DOUBAO_API_KEY` |
| Qianwen | `DASHSCOPE_API_KEY` |
| Hunyuan | `TENCENTCLOUD_SECRET_ID`, `TENCENTCLOUD_SECRET_KEY` |

Provider credentials come from the environment. Missing Doubao/Qianwen credentials raise before network dispatch. Hunyuan has distinct SDK/setup error handling and is not guaranteed to share that behavior. Existing research scripts may additionally read local ignored WorldQuant credential files. Do not add their values or saved response bodies to Git. Rotate previously exposed credentials with the provider; source-history cleanup cannot revoke them.

## Execution boundary

Several scripts authenticate, read files or launch network work at import/top level (for example `alphas_zero/day2.py`). Do not run or import them merely to inspect a checkout. There is no common offline mode or broker-safe orchestration guarantee for the miners. Actual research requires script-specific dependencies, native access/settings review and explicit execution authorization. No automatic deployment or submission schedule is configured by CI.

[Design and limitations](DESIGN.md) explains request identity, outputs, retry risks and maintenance. [Release review](RELEASE_REVIEW.md) records the 0.0.3 model-attribution fix. The nested [Pyramid Crasher reference](alphas2/README.md) is inherited strategy documentation, not proof of implemented performance or current API behavior.
