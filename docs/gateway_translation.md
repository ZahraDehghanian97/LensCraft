# Human prompts through Vercel AI Gateway

Run from the `LensCraft` directory in its Python environment. Install the
`openai`, `python-dotenv`, and `pydantic` dependencies from `requirements.txt`. Set
`AI_GATEWAY_API_KEY` in `.env`; the CLI loads that file without overriding exported
environment variables. The key is never written to annotations or provenance.

```bash
python src/translate_prompt.py \
  --prompt 'The camera trucks right while the character remains stationary.' \
  --model zai/glm-5.3-flash \
  --output outputs/prompt.json \
  --metadata-output outputs/prompt.metadata.json
```

The default model is `zai/glm-5.3-flash`, verified in the
[Gateway model catalog](https://ai-gateway.vercel.sh/v1/models).
`--model` or `AI_GATEWAY_MODEL` can explicitly select another Gateway model ID.
The translator does not configure fallback models. It requests `low` reasoning
effort for this extraction task. `--reasoning-effort` accepts the GLM catalog's
verified `low`, `high`, and `max` values; the catalog does not advertise disabled
reasoning. Vercel documents the standard
[`reasoning_effort` parameter](https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions/reasoning).

Output contains `cinematographyPrompts` and `simulationInstructions`, each with
exactly one item. Its vocabulary and types derive from the model's current
`cinematography_struct` and `simulation_struct`: 10 cinematography leaves plus
36 simulation leaves. The model is instructed to leave fields absent from the
description as `null`, which the conditioning encoder marks as missing, and to
avoid inventing numeric constraints or camera composition. Schema validation
does not guarantee semantic accuracy; the ET run found unsupported `importance`
values in 16 distinct caption translations covering 36 clips. One output
represents one continuous shot; multiple separately timed shots need separate
requests. The file supplies structured conditioning, not subject motion or a
complete simulator scene.

The Python API is usable from inference code with `PYTHONPATH=src`:

```python
from annotator.gateway import GatewayTranslator

translator = GatewayTranslator()
conditioning = translator.translate("Keep a fixed camera while the subject walks left.")
prompt = conditioning["cinematographyPrompts"][0]
instruction = conditioning["simulationInstructions"][0]
# Pass prompt and instruction to data.simulation.utils.fix_prompts_and_instructions.
```

For dataset translation, supply original natural-language captions as JSONL or
a JSON array. No previous structured annotations are read:

```json
{"id":"et-example1","prompt":"The camera pans left."}
{"id":"et-example2","prompt":"The camera slowly moves toward the character."}
```

```bash
python src/translate_prompt.py \
  --input native_captions.jsonl \
  --output outputs/fresh_annotations.json \
  --metadata-output outputs/fresh_annotations.metadata.json \
  --cache-dir outputs/gateway_cache \
  --workers 4
```

The annotation file maps each input ID to its validated conditioning envelope,
preserving input order. Successful calls are cached immediately; rerun the same
command after an interruption to resume. Cache keys include the original prompt,
model, current schema, translation instructions, output mode, base token budget,
and reasoning effort.
Old annotations and a different translation configuration cannot satisfy this
cache. Provenance records source caption, requested/returned model, response ID,
schema/instruction hashes, token usage, and timestamp. Choose a new cache
directory when intentionally requesting a fresh independent run.

Requests use Vercel's documented
[Python OpenAI-compatible API](https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions)
and [JSON schema output](https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions/structured-outputs).
Every response is also validated locally by strict Pydantic v2 models, generated
from the same canonical structures used by the encoder. Pydantic produces the
request JSON schema; the 46 field definitions are not duplicated.
Invalid enums, extra/missing fields,
wrong booleans, nonfinite/out-of-range numbers, duplicate JSON keys, refusals,
truncated output, and multiple-shot responses are rejected. Invalid JSON or schema
responses receive up to two corrective requests to the same model, with the
original invalid response, the exact validator error, and the full Pydantic
schema to repair all missing nested fields. No values are silently
coerced. `--validation-retries 0` disables correction; exhausted retries fail
without caching invalid data or dropping the sample. Provenance records attempt
count, response IDs/models, validation errors, and total token usage across all
returned attempts. A truncated completion is always discarded and retried with
twice the output budget: by default 8,192 → 16,384 → 32,768 tokens, capped at the
GLM catalog's 131,000-token limit. Truncation and validation share the same two-retry
allowance; partial JSON is never accepted. Each attempt records its budget,
reasoning effort, finish reason, and available reasoning-token count. Successful
cached translations remain reusable when only the retry limit changes. Changing
the schema or reasoning effort creates a new cache key, so the Pydantic/low-effort
protocol does not reuse earlier translations with different generation settings.
Refusals and API errors do not trigger corrective retries. If a selected
provider rejects JSON schema mode, `--response-format text` explicitly requests
plain JSON with the full schema in the system message and the same strict local
validation. It keeps the selected model. `--max-tokens` and `--timeout` control
the output budget and request timeout; SDK retries are limited to two.

Run the network-free boundary checks:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_gateway_translation.py' -v
```
