# Shokoban (書庫番, "the archive keeper")

[日本語](README.md)

Shokoban takes a transcript of a business meeting, **masks names and other identifying details entirely on your own machine**, lets **a human review the full masked text**, and only then turns that reviewed text into **progress notes**.
The goal: you don't have to remember every detail of a meeting, because you can trace it all later.

> Early software (0.1). The author uses it every day for real work and keeps improving it. It handles **Japanese-language meetings only** and **runs on macOS only** (see "Where it runs"). The gaps are listed honestly under "Known limitations". Improvements are welcome.
>
> The GUI and all prompts are in Japanese. This English page explains what the project is so you can decide whether to dig in.

## What it does

```
transcript ──▶ ① mask (rules + name dictionary + local LLM + GiNZA) ──▶ ② name review (human)
           ──▶ ③ pre-send review (human reads the entire masked text) ──▶ ④ a reader model writes the progress notes
           ──▶ ⑤ human edits and saves ──▶ ⑥ shared folder (other tools or agents read it from there)
```

Sections of a progress note (the GUI still calls them "5欄", the five sections, from before the first one was added):

- the plan or proposal we described (こちらから話した道筋・提案)
- the customer's problems (困りごと)
- what we were asked to do (頼まれていること)
- what we will do (こちらがやること)
- what was decided (決まったこと)
- what is still undecided (まだ決まっていないこと)

Each line is marked with who said it (customer or us).

## Privacy model

- Transcripts, customer names and unmasked text **never leave this machine**.
- The reader in step ④ only ever receives **masked text that a human has read in full and marked as reviewed** in step ③. Right before sending, Shokoban compares the payload's sha256 fingerprint with the reviewed one, and refuses to send if even a single character differs.
- The reader is **off by default**. Nothing is sent anywhere until you enable it with an environment variable.
- Files are written with mode 0600 and folders with 0700.

### How masking works, and its limits

- Masking is mostly **rules (regular expressions and word lists)**. The local LLM is used to find name candidates and to put a short label on each masked region. GiNZA (optional) adds more name candidates.
- Masked text is replaced with tags such as 〈名前〉 (name), 〈社名〉 (company), 〈連絡先〉 (contact), 〈金額：…〉 (amount band), 〈数字〉 (number), 〈体調の事情〉 (health matter), 〈家族の事情〉 (family matter), or 〈この文は伏せました〉 for a whole sentence.
- Person and company names are **decided by a human, one term at a time**, on screen ② from the candidates the rules and the local LLM found. Candidates you leave undecided stay masked.
- **There is no dedicated rule for place names or addresses.** Nor can a machine stop someone from guessing who the other party is from context, such as their trade or products.
- Screen ③ highlights sentences that look risky, but only up to 10% of the text. **The last gate is a human reading the full text on screen ③.** You can mask any extra sentence right there.

## What kind of meetings it is tuned for

The prompts and word lists are tuned for meetings recorded by **people who help small shops and businesses with their websites, flyers, social media, photos and other marketing work**, because that is the author's job. It assumes:

- The recording was made by the helping side ("us"), and there are two parties: the customer and us.
- The transcript needs no speaker labels (the reader infers who is speaking). Sentences are split on 。！？ and line breaks.
- Dates and month boundaries use Japan time.

Shokoban works for other kinds of meetings too, but it may pick up the wrong things. Improvements for other industries are welcome.

## Where it runs

- **macOS (Apple Silicon recommended)**, Python 3.12+.
  - On Linux the local LLM cannot be started (free memory is checked with macOS's `memory_pressure`). On Windows the GUI does not start.
- `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp), plus a GGUF model that handles Japanese well (the author uses a Japanese-tuned Gemma 4 12B).
  - **You need a recent llama.cpp.** Shokoban passes `--reasoning off` and `--no-webui` (the author uses build 12110). With an older build the server fails to start, and the GUI only says it did not respond within 120 seconds.
  - Shokoban starts it on `127.0.0.1:1234` only when needed. **The port is fixed at 1234**, so it cannot run alongside other software that uses 1234, such as LM Studio.
  - Register the model under the name `gemma-4-12b-ja` in `models` in `runtime.json`. To use a different name, set `SHOKOBAN_TAGGER_MODEL`.
  - **The paths to the model file and to `llama-server` must contain only plain ASCII with no spaces.** Shokoban needs this to recognise the process it started, and rejects other paths when it reads the settings.
- Keep the work folder (where your data lives) readable only by you (`chmod 700`), on a disk where permissions work, such as the Mac's internal disk.
- Optional: GiNZA, which adds more person and company name candidates. It runs in its own venv.
- Optional: a reader, either the [Claude Code](https://docs.anthropic.com/en/docs/claude-code) `claude` command or a local OpenAI-compatible server (see "Readers").

## Getting started

```bash
git clone https://github.com/ogd328/shokoban.git && cd shokoban
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Create a work folder and add the local LLM settings:

```bash
mkdir -p ~/shokoban-data && chmod 700 ~/shokoban-data
cp examples/runtime.example.json ~/shokoban-data/runtime.json
# edit "models" (absolute path to your GGUF) and "state_dir" (absolute path)
```

Keys in `runtime.json`: `ctx` (context length), `idle_minutes` (how long the server may sit unused), `min_free_percent` (do not start if free memory is below this), `default_model` and `models` (names and absolute GGUF paths), `state_dir` (where start-up records are kept).

Optional: install GiNZA into `venv-ginza` inside the work folder:

```bash
python3.12 -m venv ~/shokoban-data/venv-ginza
~/shokoban-data/venv-ginza/bin/pip install ginza ja_ginza
```

Open the GUI in **synthetic-data mode** first:

```bash
SHOKOBAN_WORK_DIR=~/shokoban-data SHOKOBAN_DATA_MODE=synthetic \
  .venv/bin/streamlit run gui/shokoban_mac.py \
  --server.address 127.0.0.1 --browser.gatherUsageStats false
```

`--browser.gatherUsageStats false` stops Streamlit from sending usage statistics.

Try it with the sample transcript. In synthetic mode only files whose customer name starts with 架空顧客 ("fictional customer") are imported:

```bash
cp "examples/2030-01-15 架空顧客サンプル.txt" ~/shokoban-data/inbox/
```

(The `inbox` folder is created the first time you open the GUI.)

For real data, set `SHOKOBAN_DATA_MODE=real SHOKOBAN_REVIEW_REAL=1`. Without `SHOKOBAN_REVIEW_REAL=1` the name review and pre-send review screens stay closed. Synthetic and real data live in separate folders (`store-synthetic/` and `store/`).

## How to use it

1. **Put a transcript in the inbox.** Save it in `<work folder>/inbox/` as `2026-01-15 Customer name.txt`: an ISO date, one space, the customer name; `.txt` or `.md`, UTF-8, up to 5 MB. The name part becomes the customer's name.
2. **Import.** On 案件一覧, press 受け取りフォルダを取り込む. This creates a draft meeting and moves the file to `inbox/取り込み済み/`.
3. **Name review (名前の確認).** Press 候補を出す, then decide each term (person / company / not a name / later) and save it to the dictionary.
4. **Pre-send review (送る前の確認).** Open it right after step 3 in the same browser session, because it uses the candidates from step 3. Press 伏せて確かめる, read the entire masked text, mask any extra sentence, tick the "I read all of it" box and press 確認済みにする.
5. **Make the notes (5欄を作る).** If a reader is enabled, it writes the progress notes from the reviewed text. You can edit and save them.
6. **Confirm the meeting.** On 案件ファイル, confirm the draft. Confirmed meetings feed 進行の状態 (the local LLM drafts stage, next task, due date and who moves next) and 手元AIに聞く (ask the local LLM).
   - In synthetic mode, confirmed meetings no longer appear on screens 3–5, so finish those first.
7. **Share.** Once you set a share label under 進行の状態, you can write the notes to `<work folder>/share/進行メモ/` (`share-synthetic/` in synthetic mode). Let other tools and agents read only that folder.

Do not edit the stored files by hand. They carry a fingerprint of their contents, and loading stops if it does not match.

## Readers

Choose one with `SHOKOBAN_CLOUD_READER`. Left empty, the reader is off.

- `claude`: runs the Claude Code CLI (`claude -p --model <model> --tools "" --output-format json`).
  - `claude` must be on PATH and logged in. Cost depends on your own plan.
  - `claude` loads your own settings (CLAUDE.md, hooks, MCP servers). **If you have hooks or MCP servers that reach outside, the masked text may be passed to them too.**
  - On failure the GUI shows only the exit code, whatever the cause (not logged in, old version, model not available).
  - Calls are capped at `SHOKOBAN_READER_MONTHLY_LIMIT` per month (default 200).
- `local` (experimental): sends to an OpenAI-compatible server on this machine (`SHOKOBAN_READER_URL`, default `http://127.0.0.1:1234/v1`). Other hosts are refused.
  - You must start the server yourself. The reader prompt alone is about 7,000 Japanese characters, plus the meeting text, so it needs **a large context (tens of thousands of tokens)**. It does not fit in the example `ctx: 8192`.
  - Whether a small model can return the required JSON is untested. If the reply is malformed, the notes come out empty.
- `fake`: returns a fixed shape locally, for trying the screens.

## Stopping the local LLM automatically

The GUI starts llama-server when it needs it, but stopping it is a separate small script. Run `scripts/llm_idle_stop.py` once a minute and it stops a llama-server that Shokoban started once it has been unused for `idle_minutes`. A macOS example is in `deploy/macos-launchd/shokoban-llm-idle-stop.example.plist`. Without it, stop the server with 手元AIを今すぐ止める on the 手元AI screen; until then it keeps using memory.

An example for keeping the GUI running is in `deploy/macos-launchd/shokoban-gui.example.plist`.

## Configuration (environment variables)

| Name | Default | Meaning |
|---|---|---|
| `SHOKOBAN_WORK_DIR` | (required) | work folder (absolute path); `runtime.json` is read from here |
| `SHOKOBAN_DATA_MODE` | (required) | `real` or `synthetic` |
| `SHOKOBAN_REVIEW_REAL` | empty | `1` opens the name review and pre-send review screens for real data |
| `SHOKOBAN_LLAMA_SERVER` | `llama-server` on PATH | path to llama-server |
| `SHOKOBAN_TAGGER_MODEL` | `gemma-4-12b-ja` | name of the local model (a key in `models` in `runtime.json`) |
| `SHOKOBAN_CLOUD_READER` | empty (off) | reader: `claude` / `local` / `fake` |
| `SHOKOBAN_CLAUDE_MODEL` | `opus` | model for `claude` |
| `SHOKOBAN_READER_URL` | `http://127.0.0.1:1234/v1` | endpoint for `local` (this machine only) |
| `SHOKOBAN_READER_MODEL` | none | model name for `local` |
| `SHOKOBAN_READER_MONTHLY_LIMIT` | `200` | maximum reader calls per month |

## Known limitations

- macOS only (see above). Port 1234 cannot be changed.
- When llama-server fails to start, the GUI does not show why (arguments, model or memory).
- The GUI and prompts are Japanese only. Some error messages are in English.
- There is no screen for registering other names for a customer (the name dictionary covers this).
- Many limits and thresholds are hard-coded, such as highlighting at most 10% of the text and reader quotes of at most 120 characters.
- Some module names (`pick_bench`, `mask_stage5`, `r3`) are leftovers from development.

## Tests

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q
```

The tests use only synthetic data with a fake local LLM and a fake reader, and never touch the network.

## Contributing

All kinds of improvements are welcome: fewer masking misses, support for other operating systems, a reader that runs fully locally, support for other industries, a friendlier UI. Issues and pull requests can be in English or Japanese.

**Never put real meeting text or the names of real people or companies in issues or test data.** Please use made-up examples.

## License

[Apache License 2.0](LICENSE) — Copyright 2026 OgD
