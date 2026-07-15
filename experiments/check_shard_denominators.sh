#!/usr/bin/env bash
# Prove that gpt2_baseline.py --shard agrees with a shard `quark prepare` really
# wrote, on a real file, through the real CLI.
#
# The fixture proves the Rust and the Python agree on toy strings. It cannot
# prove they agree on the corpus, because it does not know what `quark prepare`
# does to a file -- and that gap is where the bug this guards against lived: both
# sides counted words and bytes identically and still disagreed, because one
# summed them per document and the other counted them on the whole file. The
# numbers matched on every fixture case and diverged on every real corpus.
#
# So this runs the actual binary over an actual file and compares the sidecar to
# what the Python computes independently. Cheap: no model, no GPU, ~5 seconds.
#
#   ./experiments/check_shard_denominators.sh
set -euo pipefail

cd "$(dirname "$0")/.."
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Shaped like the real wiki.*.tokens: leading blank line, space-padded article
# headings, section headings that must not split, blank lines between articles.
cat > "$work/corpus.txt" <<'EOF'

 = Valkyria Chronicles III =

 Senjō no Valkyria 3 is a tactical role @-@ playing game developed by Sega .

 = = Gameplay = =

 The game is a tactical RPG played from a top @-@ down perspective .

 = Tower Building =

 The Tower Building of the Little Rock Arsenal is a building in Arkansas .

 = Cicely Mary Barker =

 Cicely Mary Barker was an English illustrator best known for fairy paintings .
EOF

echo "==> quark tokenizer"
cargo run --quiet --bin quark -- tokenizer "$work/corpus.txt" \
  --vocab-size 512 --out "$work/tok.json"

echo "==> quark prepare --split-articles"
cargo run --quiet --bin quark -- prepare "$work/corpus.txt" \
  --out "$work/corpus.bin" --tokenizer "$work/tok.json" --split-articles

echo "==> comparing the sidecar to what gpt2_baseline.py counts independently"
python3 - "$work/corpus.txt" "$work/corpus.bin" <<'PY'
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, "experiments")
from gpt2_baseline import check_denominators, corpus_denominators, documents

text_path, shard_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
text = text_path.read_text(encoding="utf-8")
meta = json.loads(shard_path.with_suffix(".json").read_text())

docs = documents(text, split_articles=True)
print(f"    {len(docs)} documents, shard has {meta['n_tokens']} tokens")

# The check the README's command performs. Raises SystemExit on a mismatch.
check_denominators(corpus_denominators(docs), shard_path)

# ...and the failure it exists to catch: the same counters, run over the whole
# file instead of the documents. If this does NOT differ, the check above is
# passing for free and proves nothing on this corpus.
whole = (len(text.split()), len(text.encode("utf-8")))
ours = corpus_denominators(docs)
assert whole != ours, (
    f"whole-file counts {whole} equal per-document counts {ours}, so this corpus "
    f"cannot distinguish the two. The check above is vacuous here."
)
print(f"    counting on the whole file instead would have said {whole[0]} words / "
      f"{whole[1]} bytes")
print(f"    -- a {whole[1] - ours[1]} byte difference, which is the bug this catches")
PY

echo "==> ok"
