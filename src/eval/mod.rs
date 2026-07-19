//! Evaluation: the numbers the issue actually asks for.
//!
//! Three of them, and they answer different questions:
//!
//! * [`corpus`] -- perplexity on WikiText-103, in word-level and bits-per-byte
//!   units. The headline claim, and the only one with a GPT-2 number to beat.
//! * [`blimp`] -- grammatical knowledge, as a forced choice between a minimal
//!   pair. Perplexity can be bought with frequency statistics; BLiMP cannot.
//! * [`generate`] -- a fixed prompt set, decoded deterministically. Neither of
//!   the above notices if the model has learned to produce fluent text and
//!   nothing else, and both are invisible to a human reader.
//!
//! Everything here takes `B: Backend`, not `AutodiffBackend`: evaluation needs
//! no gradients, and asking for them would double the memory for nothing.

pub mod blimp;
pub mod corpus;
pub mod generate;

use std::{path::PathBuf, sync::Arc};

use anyhow::{bail, Context, Result};
use burn::{
    module::Module,
    prelude::Backend,
    record::{CompactRecorder, FileRecorder},
};

pub use blimp::{BlimpScore, BlimpSuite};
pub use corpus::{evaluate, CorpusScore, EvalConfig};
pub use generate::{GenerationConfig, Sample};

use crate::{
    compress::{
        eval::{load_compressor, CompressEvalConfig},
        CompressTrainConfig,
    },
    data::{tokenizer, Shard},
    model::QuarkLm,
    TrainConfig,
};

/// Rebuild a trained model from a run's artifact directory.
///
/// The architecture comes from the run's own `config.json`, not from a flag. A
/// record carries only tensors, so the config is the sole thing that says what
/// shape they were -- and taking it from the run that wrote them is the only way
/// to be sure.
///
/// # The check at the end is not defensive padding
///
/// burn's recorder does **not** validate shapes on load. Loading a `d_model=64`
/// record into a `d_model=128` model returns `Ok`, with the record's tensors
/// installed and the config's shape silently discarded. That is verified by
/// `loading_a_record_whose_shape_contradicts_the_config_is_an_error` below: it
/// asserted an error and got a model back.
///
/// Left alone, a stale `config.json` would produce a model that is neither the
/// trained one nor the configured one, and it would report a perplexity rather
/// than a failure. Comparing parameter counts costs one traversal and turns that
/// into an error at load time.
pub fn load_model<B: Backend>(
    config_path: &std::path::Path,
    model_path: &std::path::Path,
    device: &B::Device,
) -> Result<(QuarkLm<B>, TrainConfig)> {
    let extension = <CompactRecorder as FileRecorder<B>>::file_extension();
    let record = model_path.with_extension(extension);
    if !record.is_file() {
        bail!("{}", missing_record_message(model_path, &record, extension));
    }
    let train_config = TrainConfig::load(config_path)
        .with_context(|| format!("reading the run's config {}", config_path.display()))?;
    let model = QuarkLm::<B>::new(train_config.model.clone(), device)
        .load_file(model_path, &CompactRecorder::new(), device)
        .with_context(|| format!("loading weights from {}", model_path.display()))?;

    let loaded = model.num_params();
    let expected = train_config.model.param_count();
    if loaded != expected {
        bail!(
            "the record at {} holds {loaded} parameters but {} describes a {expected}-parameter \
             model: the config does not belong to these weights. burn's recorder installs the \
             record's shapes without complaint, so this would otherwise be a wrong number rather \
             than an error",
            model_path.display(),
            config_path.display(),
        );
    }

    Ok((model, train_config))
}

/// What to say when the record `--model` names is not there.
///
/// burn appends the recorder's extension itself, so `--model foo` reads
/// `foo.mpk` and the path in the error is not the path the user typed. Left to
/// burn, the whole diagnosis is `FileNotFound("No such file or directory")` --
/// which is what a run against a *directory* of checkpoints produced on the
/// maintainer's machine (PR #11), costing four attempts to guess the spelling
/// the flag wanted. Everything needed to fix it is on disk, so name it: the file
/// that was looked for, and the records that do exist nearby.
pub(crate) fn missing_record_message(
    model_path: &std::path::Path,
    record: &std::path::Path,
    extension: &str,
) -> String {
    let mut msg = if model_path.is_dir() {
        format!(
            "{} is a directory; `--model` wants the record itself, as a path without the \
             .{extension} extension (the recorder sets it)",
            model_path.display(),
        )
    } else {
        format!(
            "no model record at {} (`--model` is a path *without* the .{extension} extension, \
             which the recorder sets)",
            record.display(),
        )
    };
    // A directory holds candidates; a file path's siblings are the candidates.
    let dir = if model_path.is_dir() {
        model_path.to_path_buf()
    } else {
        model_path.parent().unwrap_or(model_path).to_path_buf()
    };
    let mut found = records_in(&dir, extension);
    found.extend(records_in(&dir.join("checkpoint"), extension));
    if !found.is_empty() {
        msg += "\n\nrecords that do exist (pass one of these to --model):";
        for path in found {
            msg += &format!("\n  {}", path.display());
        }
    }
    msg
}

/// The loadable records in `dir`, named the way `--model` wants them: with the
/// extension stripped back off.
fn records_in(dir: &std::path::Path, extension: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<_> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == extension))
        // Optimiser and scheduler state are recorded next to the weights and are
        // not loadable as a model, so offering them would just be a second wrong
        // guess.
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("model"))
        })
        .map(|p| p.with_extension(""))
        .collect();
    found.sort();
    found
}

/// Everything `quark eval` needs to know.
#[derive(Debug, Clone)]
pub struct EvalRun {
    pub artifact_dir: PathBuf,
    /// Defaults to `<artifact_dir>/config.json`.
    pub config_path: Option<PathBuf>,
    /// Defaults to `<artifact_dir>/model`.
    pub model_path: Option<PathBuf>,
    pub tokenizer_path: PathBuf,
    /// Shard to measure perplexity on. `None` skips the corpus sweep.
    pub shard: Option<PathBuf>,
    /// BLiMP `.jsonl` directory. `None` skips BLiMP.
    pub blimp_dir: Option<PathBuf>,
    pub generate: bool,
    pub corpus: EvalConfig,
    pub generation: GenerationConfig,
    pub blimp_batch_size: usize,
    /// How to sweep a *compressor* run, when the config turns out to describe
    /// one. Ignored otherwise.
    pub compress: CompressEvalConfig,
    /// Spans to round-trip and print under `--generate`, for a compressor run.
    pub compress_samples: usize,
}

/// Whether `path` holds a [`CompressTrainConfig`] rather than a [`TrainConfig`].
///
/// Decided by the presence of the `compress` key rather than by trying both
/// parsers, because "which config is this" deserves a better answer than the
/// second parser's error. It is also the failure the maintainer actually hit
/// (issue #14): a compressor run's `config.json` fed to `quark eval` reported
/// `missing field 'model' at line 74`, which names a field that *is* there --
/// nested one level down, in a config of the other kind.
fn is_compress_config(path: &std::path::Path) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|v| v.get("compress").map(|c| c.is_object()))
        .unwrap_or(false)
}

/// Evaluate a compressor run: reconstruction, the rate, and sample round trips.
///
/// Shares the flags of the language-model path deliberately -- `--ppl` names
/// the shard to measure on and `--generate` asks for text a human can read --
/// because they mean the same thing here. What differs is what is measured on
/// them, and [`crate::compress::eval`] documents why those are the numbers.
fn run_compressor<B: Backend>(
    run: &EvalRun,
    config_path: &std::path::Path,
    model_path: &std::path::Path,
    device: &B::Device,
) -> Result<String> {
    use crate::compress::eval as ceval;

    let config = CompressTrainConfig::load(config_path)
        .with_context(|| format!("reading the run's config {}", config_path.display()))?;
    let model = load_compressor::<B>(&config, model_path, device)?;
    let c = &config.compress;

    let mut out = format!(
        "compressor {} ({} parameters, {} tokens -> {} slots)\n\n{}\n",
        model_path.display(),
        c.param_count(),
        c.span_len,
        c.n_slots,
        c.budget_table(),
    );

    // BLiMP scores a language model's preference between two sentences, and a
    // compressor has no next-token distribution to prefer with. Saying so beats
    // ignoring the flag, which would look like a suite that scored 0%.
    if run.blimp_dir.is_some() {
        out += "\n== BLiMP ==\nskipped: BLiMP is a language-model evaluation and this run is a \
                compressor, which has no next-token distribution to score a minimal pair with\n";
    }

    if let Some(path) = &run.shard {
        let shard = Arc::new(
            Shard::open(path).with_context(|| format!("opening shard {}", path.display()))?,
        );
        let score = ceval::evaluate(&model, shard, &run.compress, device)?;
        out += &format!(
            "\n== reconstruction: {} ==\n{}\n",
            path.display(),
            score.report()
        );
    } else {
        out += "\nno --ppl shard given, so nothing was reconstructed; pass the test shard to \
                measure this compressor\n";
    }

    if run.generate {
        let Some(path) = &run.shard else {
            bail!(
                "--generate needs text to round-trip, and a compressor has no prompt set of its \
                 own: pass --ppl <shard> as well"
            );
        };
        let tok = tokenizer::load(&run.tokenizer_path)?;
        let shard = Arc::new(Shard::open(path)?);
        let samples = ceval::samples(&model, &tok, shard, run.compress_samples, device)?;
        out += &format!("\n== round trips ==\n{}", ceval::report(&samples));
    }

    Ok(out)
}

/// Run whichever evaluations were asked for and return the report as text.
///
/// Generic over the backend for the same reason [`crate::train::run`] is: the
/// tests exercise this exact function on the CPU, so the code path the
/// maintainer runs on a GPU is the code path CI checks.
pub fn run<B: Backend>(run: &EvalRun, device: &B::Device) -> Result<String> {
    let config_path = run
        .config_path
        .clone()
        .unwrap_or_else(|| run.artifact_dir.join("config.json"));
    let model_path = run
        .model_path
        .clone()
        .unwrap_or_else(|| run.artifact_dir.join("model"));

    // A compressor run writes a config of a different shape, and `quark eval`
    // is the command a user reaches for either way. Dispatching on the file
    // rather than on a flag means the obvious invocation works instead of
    // failing on a field that is present but nested (issue #14).
    if is_compress_config(&config_path) {
        return run_compressor::<B>(run, &config_path, &model_path, device);
    }

    let (model, train_config) = load_model::<B>(&config_path, &model_path, device)?;
    let mut out = format!(
        "model {} ({} parameters, {} compute-equivalent)\n\n",
        model_path.display(),
        train_config.model.param_count(),
        train_config.model.compute_equivalent_params(),
    );

    if let Some(path) = &run.shard {
        let shard = Arc::new(
            Shard::open(path).with_context(|| format!("opening shard {}", path.display()))?,
        );
        let score = corpus::evaluate(&model, shard, &run.corpus, device)?;
        out += &format!("== corpus: {} ==\n{}\n\n", path.display(), score.report());
    }

    if run.blimp_dir.is_some() || run.generate {
        let tok = tokenizer::load(&run.tokenizer_path)?;

        if let Some(dir) = &run.blimp_dir {
            let suite = BlimpSuite::load(dir)?;
            tracing::info!(
                paradigms = suite.paradigms.len(),
                pairs = suite.n_pairs(),
                "scoring BLiMP"
            );
            let score = blimp::evaluate(&model, &tok, &suite, run.blimp_batch_size, device)?;
            out += &format!("== BLiMP ==\n{}\n", score.report());
        }

        if run.generate {
            let samples = generate::run_suite(&model, &tok, &run.generation, device)?;
            out += &format!("== generation ==\n{}", generate::report(&samples));
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::{tokenizer::EOS_TOKEN, ShardWriter},
        test_util::TestBackend,
        ModelConfig,
    };
    use std::{io::Write, path::Path};

    /// A complete artifact directory: a tokenizer, a shard, a saved model, and
    /// the `config.json` that describes it.
    fn artifacts(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
        let text = dir.join("corpus.txt");
        let mut f = std::fs::File::create(&text).unwrap();
        for i in 0..300 {
            writeln!(
                f,
                "the city of number {i} begins with a bridge and three islands"
            )
            .unwrap();
        }
        drop(f);

        let tok_path = dir.join("tokenizer.json");
        let tok = tokenizer::train(&[text.to_str().unwrap().to_string()], 256, &tok_path).unwrap();
        let vocab = tok.get_vocab_size(true);

        let shard_path = dir.join("valid.bin");
        let mut w =
            ShardWriter::create(&shard_path, vocab, tokenizer::eos_id(&tok).unwrap()).unwrap();
        let body = std::fs::read_to_string(&text).unwrap();
        w.push_document(&body, &tokenizer::encode(&tok, &body).unwrap())
            .unwrap();
        w.finish().unwrap();

        let model_cfg = ModelConfig {
            vocab_size: vocab,
            // Not `tiny()`'s 64. This tokenizer is trained on a 300-line corpus,
            // so it has no merges for most of `generate::PROMPTS` and encodes
            // them close to byte-per-token -- the longest reaches 64 exactly, and
            // a prompt that fills the context is (correctly) an error. The real
            // model pairs an 8192-entry vocabulary with a 512-token context.
            max_seq_len: 256,
            ..ModelConfig::tiny()
        };
        let train_cfg = TrainConfig {
            model: model_cfg.clone(),
            ..Default::default()
        };
        let config_path = dir.join("config.json");
        train_cfg.save(&config_path).unwrap();

        let model_path = dir.join("model");
        QuarkLm::<TestBackend>::new(model_cfg, &Default::default())
            .save_file(&model_path, &CompactRecorder::new())
            .unwrap();

        (config_path, model_path, shard_path)
    }

    fn eval_run(dir: &Path, config_path: PathBuf, model_path: PathBuf) -> EvalRun {
        EvalRun {
            artifact_dir: dir.to_path_buf(),
            config_path: Some(config_path),
            model_path: Some(model_path),
            tokenizer_path: dir.join("tokenizer.json"),
            shard: None,
            blimp_dir: None,
            generate: false,
            corpus: EvalConfig {
                seq_len: 32,
                stride: 16,
                batch_size: 4,
                num_workers: 1,
            },
            generation: GenerationConfig {
                max_new_tokens: 4,
                ..Default::default()
            },
            blimp_batch_size: 4,
            compress: CompressEvalConfig {
                batch_size: 2,
                num_workers: 1,
                max_spans: None,
            },
            compress_samples: 2,
        }
    }

    /// A saved model must come back identical. Loading is the step between
    /// "training worked" and "the number in the report is this model's" -- if it
    /// silently loaded a fresh model instead, every evaluation would report the
    /// initialization and look merely disappointing rather than broken.
    #[test]
    fn a_loaded_model_scores_exactly_what_it_scored_before_saving() {
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();
        let (config_path, model_path, shard_path) = artifacts(dir.path());
        let run = eval_run(dir.path(), config_path.clone(), model_path.clone());

        let (original, _) = load_model::<TestBackend>(&config_path, &model_path, &device).unwrap();
        let shard = Arc::new(Shard::open(&shard_path).unwrap());
        let before = corpus::evaluate(&original, shard.clone(), &run.corpus, &device).unwrap();

        // Round-trip through the recorder a second time, exactly as `quark eval`
        // would after a real run.
        let (reloaded, _) = load_model::<TestBackend>(&config_path, &model_path, &device).unwrap();
        let after = corpus::evaluate(&reloaded, shard, &run.corpus, &device).unwrap();

        assert!(
            (before.total_nll - after.total_nll).abs() < 1e-3,
            "the same weights scored {} and then {}",
            before.total_nll,
            after.total_nll
        );
    }

    /// The whole CLI path in one test: load, sweep a corpus, score BLiMP, decode
    /// the prompt set. It asserts wiring, not quality -- an untrained model has
    /// no quality to assert.
    #[test]
    fn the_full_evaluation_runs_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let (config_path, model_path, shard_path) = artifacts(dir.path());

        let blimp_dir = dir.path().join("blimp");
        std::fs::create_dir(&blimp_dir).unwrap();
        let mut f = std::fs::File::create(blimp_dir.join("p.jsonl")).unwrap();
        writeln!(
            f,
            r#"{{"sentence_good": "the city begins", "sentence_bad": "the city begin", "UID": "agr", "field": "morphology"}}"#
        )
        .unwrap();
        drop(f);

        let mut spec = eval_run(dir.path(), config_path, model_path);
        spec.shard = Some(shard_path);
        spec.blimp_dir = Some(blimp_dir);
        spec.generate = true;

        let report = run::<TestBackend>(&spec, &Default::default()).unwrap();

        assert!(report.contains("word perplexity"), "{report}");
        assert!(report.contains("BLiMP accuracy"), "{report}");
        assert!(report.contains("== generation =="), "{report}");
        // The separator is an artifact of our own document layout; a reader
        // should never see it in a sample.
        assert!(!report.contains(EOS_TOKEN), "{report}");
    }

    /// The config is what tells the loader the model's shape. Pointing at a
    /// record that contradicts it must fail loudly.
    ///
    /// This test is why [`load_model`] counts parameters: written first, it
    /// asserted an error and burn handed back a model instead. The recorder
    /// installs whatever shapes the record holds and never consults the ones the
    /// module was built with.
    #[test]
    fn loading_a_record_whose_shape_contradicts_the_config_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let (config_path, model_path, _) = artifacts(dir.path());

        // A config claiming a wider model than the record holds.
        let mut cfg = TrainConfig::load(&config_path).unwrap();
        cfg.model.d_model = 128;
        cfg.model.d_ff = 256;
        let wrong = dir.path().join("wrong.json");
        cfg.save(&wrong).unwrap();

        let err = load_model::<TestBackend>(&wrong, &model_path, &Default::default())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("does not belong to these weights"),
            "got: {err}"
        );
    }

    /// Pointing `--model` at a directory of checkpoints is the natural first
    /// guess -- it is what happened on the maintainer's machine (PR #11), and
    /// burn answered `FileNotFound("No such file or directory")`, which names
    /// neither the file it wanted nor the ones sitting right there.
    #[test]
    fn a_directory_of_checkpoints_is_answered_with_the_checkpoints_in_it() {
        let dir = tempfile::tempdir().unwrap();
        let (config_path, _, _) = artifacts(dir.path());
        let checkpoints = dir.path().join("checkpoint");
        std::fs::create_dir_all(&checkpoints).unwrap();
        for name in [
            "model-3.mpk",
            "model-4.mpk",
            "optim-4.mpk",
            "scheduler-4.mpk",
        ] {
            std::fs::write(checkpoints.join(name), []).unwrap();
        }

        let err = load_model::<TestBackend>(&config_path, &checkpoints, &Default::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("is a directory"), "got: {err}");
        assert!(err.contains("checkpoint/model-3"), "got: {err}");
        assert!(err.contains("checkpoint/model-4"), "got: {err}");
        // Suggesting the optimiser state would just be the next wrong guess, and
        // the extension is the part the flag does not want.
        assert!(!err.contains("optim"), "got: {err}");
        assert!(!err.contains("model-4.mpk"), "got: {err}");
    }

    /// The default `--model artifacts/run/model` misses by a whole run
    /// directory, and the record that exists is one level away.
    #[test]
    fn a_missing_record_names_the_file_it_looked_for_and_its_neighbours() {
        let dir = tempfile::tempdir().unwrap();
        let (config_path, _, _) = artifacts(dir.path());

        let err =
            load_model::<TestBackend>(&config_path, &dir.path().join("mdoel"), &Default::default())
                .unwrap_err()
                .to_string();
        assert!(err.contains("mdoel.mpk"), "got: {err}");
        assert!(err.contains("the .mpk extension"), "got: {err}");
        // `artifacts()` saved one, as a real run would.
        assert!(err.contains("model"), "got: {err}");
    }

    /// Naming the record with its extension is what the flag's help says not to
    /// do, but `set_extension` makes it work -- and an error for a file that is
    /// plainly there would be indefensible.
    #[test]
    fn spelling_out_the_extension_still_loads() {
        let dir = tempfile::tempdir().unwrap();
        let (config_path, model_path, _) = artifacts(dir.path());
        load_model::<TestBackend>(
            &config_path,
            &model_path.with_extension("mpk"),
            &Default::default(),
        )
        .unwrap();
    }
}
