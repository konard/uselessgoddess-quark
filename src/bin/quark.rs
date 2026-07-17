//! The quark CLI.
//!
//! Corpus preparation is offline and explicit: you train a tokenizer once, then
//! bake each split into a shard. Both artifacts are inputs to training and to
//! evaluation, and both are content-addressed by nothing at all -- so the shard
//! metadata records the vocabulary size, and the loader refuses a shard whose
//! vocabulary disagrees with the model's.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use quark::{
    data,
    eval::{EvalConfig, EvalRun, GenerationConfig},
    TrainConfig,
};

#[derive(Parser)]
#[command(name = "quark", about = "A 3M-parameter language model family on burn")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Which backend to train on.
///
/// Every variant is always offered, even when it was not compiled in, because
/// `--backend wgpu` failing with "rebuild with `--features wgpu`" is a better
/// answer than clap reporting that the value does not exist.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum BackendChoice {
    /// CPU. Correct, and far too slow for a real run -- for smoke tests.
    Ndarray,
    /// The primary backend: Vulkan/Metal/DX12 via wgpu.
    Wgpu,
    /// CUDA, if you have it. Faster than wgpu on NVIDIA, and less portable.
    Cuda,
    /// wgpu again, compiled through SPIR-V rather than WGSL. On AMD this is
    /// usually the fastest thing here; see `--backend rocm`.
    Vulkan,
    /// ROCm/HIP, for AMD cards. Linux only, and untested by us.
    Rocm,
}

impl Default for BackendChoice {
    fn default() -> Self {
        // wgpu first: it is what the project targets, and what a 16GB consumer
        // card will actually be driven with. ndarray is the fallback only
        // because a build with no GPU feature has nothing else.
        //
        // vulkan outranks rocm on AMD for the reason given in Cargo.toml, and
        // both outrank ndarray, which is not a real answer for training.
        if cfg!(feature = "wgpu") {
            Self::Wgpu
        } else if cfg!(feature = "cuda") {
            Self::Cuda
        } else if cfg!(feature = "vulkan") {
            Self::Vulkan
        } else if cfg!(feature = "rocm") {
            Self::Rocm
        } else {
            Self::Ndarray
        }
    }
}

/// Define a training entry point per backend, and a matching stub for builds
/// that left that backend out.
///
/// The stub is the point: without it, `--backend cuda` on a wgpu build would
/// either fail to compile or silently run somewhere else. The type in the
/// invocation is only tokens until the feature is on, so naming `burn_cuda`
/// here does not require the crate to be present.
macro_rules! backend_entry {
    ($name:ident, $feature:literal, $backend:ty) => {
        #[cfg(feature = $feature)]
        fn $name(config: TrainConfig) -> Result<()> {
            quark::train::run::<burn::backend::Autodiff<$backend>>(config, Default::default())?;
            Ok(())
        }

        #[cfg(not(feature = $feature))]
        fn $name(_config: TrainConfig) -> Result<()> {
            bail!(
                "this binary was built without the `{feature}` backend; rebuild with \
                 `cargo build --release --features {feature}`",
                feature = $feature,
            )
        }
    };
}

backend_entry!(train_ndarray, "ndarray", burn_ndarray::NdArray<f32>);
backend_entry!(train_wgpu, "wgpu", burn_wgpu::Wgpu);
backend_entry!(train_cuda, "cuda", burn_cuda::Cuda);
backend_entry!(train_vulkan, "vulkan", burn_wgpu::Vulkan);
backend_entry!(train_rocm, "rocm", burn_rocm::Rocm);

/// The same, for evaluation.
///
/// Note the absence of `Autodiff`: evaluation computes no gradients, and
/// wrapping the backend anyway would have every forward pass build a tape it
/// then throws away -- roughly doubling the memory for nothing.
macro_rules! eval_entry {
    ($name:ident, $feature:literal, $backend:ty) => {
        #[cfg(feature = $feature)]
        fn $name(run: &EvalRun) -> Result<String> {
            quark::eval::run::<$backend>(run, &Default::default())
        }

        #[cfg(not(feature = $feature))]
        fn $name(_run: &EvalRun) -> Result<String> {
            bail!(
                "this binary was built without the `{feature}` backend; rebuild with \
                 `cargo build --release --features {feature}`",
                feature = $feature,
            )
        }
    };
}

eval_entry!(eval_ndarray, "ndarray", burn_ndarray::NdArray<f32>);
eval_entry!(eval_wgpu, "wgpu", burn_wgpu::Wgpu);
eval_entry!(eval_cuda, "cuda", burn_cuda::Cuda);
eval_entry!(eval_vulkan, "vulkan", burn_wgpu::Vulkan);
eval_entry!(eval_rocm, "rocm", burn_rocm::Rocm);

#[derive(Subcommand)]
enum Command {
    /// Train a byte-level BPE tokenizer on raw text.
    Tokenizer {
        /// Text files to train on. Usually just the training split -- training a
        /// tokenizer on validation or test text leaks them into the vocabulary.
        #[arg(required = true)]
        input: Vec<PathBuf>,
        #[arg(long, default_value = "artifacts/tokenizer.json")]
        out: PathBuf,
        #[arg(long, default_value_t = 8192)]
        vocab_size: usize,
    },
    /// Tokenize a text split into a shard for training or evaluation.
    Prepare {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "artifacts/tokenizer.json")]
        tokenizer: PathBuf,
        /// Split the input on WikiText ` = Article = ` headings so each article
        /// becomes its own document. Off for corpora that are already one
        /// document per line or per file.
        #[arg(long)]
        split_articles: bool,
    },
    /// Train a model on prepared shards.
    Train {
        /// A `TrainConfig` JSON, as written to `<artifact-dir>/config.json` by a
        /// previous run. Omit for the built-in reference config: `quark_3m` on
        /// `artifacts/{train,valid}.bin`.
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, value_enum)]
        backend: Option<BackendChoice>,

        // Overrides. These are the fields worth reaching for from a shell; the
        // rest of `TrainConfig` (the AdamW constants, the schedule shape) is
        // deliberately file-only, because changing them from a flag and not
        // recording why is how a run becomes unreproducible.
        #[arg(long)]
        train_shard: Option<PathBuf>,
        #[arg(long)]
        valid_shard: Option<PathBuf>,
        #[arg(long)]
        artifact_dir: Option<PathBuf>,
        #[arg(long)]
        seq_len: Option<usize>,
        /// Windows per forward pass. Lower this first when VRAM runs out, and
        /// raise `--grad-accumulation` by the same factor to keep the optimizer
        /// seeing an identical batch.
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long)]
        grad_accumulation: Option<usize>,
        #[arg(long)]
        num_epochs: Option<usize>,
        #[arg(long)]
        lr: Option<f64>,
        #[arg(long)]
        seed: Option<u64>,
        /// Resume from an epoch checkpoint in `artifact_dir`.
        #[arg(long)]
        resume_from_epoch: Option<usize>,
        /// Print the resolved config and the parameter budget, then exit without
        /// touching the GPU.
        #[arg(long)]
        dry_run: bool,
    },
    /// Evaluate a trained model: corpus perplexity, BLiMP, and generation.
    ///
    /// Each evaluation is opt-in, because they cost very different amounts: a
    /// corpus sweep is minutes, BLiMP is 134k short forward passes, and
    /// generation is seconds. With no `--ppl`, `--blimp` or `--generate`, this
    /// only loads the model and reports its budget.
    Eval {
        /// The run to evaluate. `config.json` and `model.mpk` are read from here
        /// unless overridden.
        #[arg(long, default_value = "artifacts/run")]
        artifact_dir: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        /// Path to the record, *without* the `.mpk` extension, as burn writes it.
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value = "artifacts/tokenizer.json")]
        tokenizer: PathBuf,
        /// Shard to measure perplexity on -- the test split, not the one trained
        /// on.
        #[arg(long, value_name = "SHARD")]
        ppl: Option<PathBuf>,
        /// Directory of BLiMP `.jsonl` files, from
        /// <https://github.com/alexwarstadt/blimp>.
        #[arg(long, value_name = "DIR")]
        blimp: Option<PathBuf>,
        /// Decode the fixed prompt set.
        #[arg(long)]
        generate: bool,
        #[arg(long, default_value_t = 512)]
        seq_len: usize,
        /// How far the evaluation window advances. Defaults to half of
        /// `--seq-len`, so every token is scored with at least that much
        /// context. Set it equal to `--seq-len` for a cheaper, pessimistic
        /// sweep -- but then do not compare the result to a strided one.
        #[arg(long)]
        stride: Option<usize>,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        /// `0` decodes greedily, which is deterministic and needs no seed.
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, value_enum)]
        backend: Option<BackendChoice>,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    match Cli::parse().command {
        Command::Tokenizer {
            input,
            out,
            vocab_size,
        } => {
            let files = input
                .iter()
                .map(|p| {
                    p.to_str()
                        .map(str::to_string)
                        .ok_or_else(|| anyhow::anyhow!("path is not UTF-8: {}", p.display()))
                })
                .collect::<Result<Vec<_>>>()?;
            let tok = data::tokenizer::train(&files, vocab_size, &out)?;
            tracing::info!(
                vocab = tok.get_vocab_size(true),
                "wrote tokenizer to {}",
                out.display()
            );
        }
        Command::Prepare {
            input,
            out,
            tokenizer,
            split_articles,
        } => {
            if !tokenizer.exists() {
                bail!(
                    "no tokenizer at {}; run `quark tokenizer` first",
                    tokenizer.display()
                );
            }
            let tok = data::tokenizer::load(&tokenizer)?;
            data::prepare_shard(&input, &out, &tok, split_articles)?;
        }
        Command::Train {
            config,
            backend,
            train_shard,
            valid_shard,
            artifact_dir,
            seq_len,
            batch_size,
            grad_accumulation,
            num_epochs,
            lr,
            seed,
            resume_from_epoch,
            dry_run,
        } => {
            let mut cfg = match &config {
                Some(path) => TrainConfig::load(path)
                    .with_context(|| format!("reading config {}", path.display()))?,
                None => TrainConfig::default(),
            };

            // `resume_from_epoch` is deliberately not `Option`-overwritten the
            // way the rest are: `None` from the CLI means "say nothing", which
            // must not clear a value the file set.
            macro_rules! set {
                ($($field:ident),* $(,)?) => { $( if let Some(v) = $field { cfg.$field = v; } )* };
            }
            set!(
                train_shard,
                valid_shard,
                artifact_dir,
                seq_len,
                batch_size,
                grad_accumulation,
                num_epochs,
                lr,
                seed,
            );
            if resume_from_epoch.is_some() {
                cfg.resume_from_epoch = resume_from_epoch;
            }

            // Validate before dispatching, so a typo is caught here rather than
            // after a backend has spun up a GPU context.
            cfg.validate()?;

            let backend = backend.unwrap_or_default();
            if dry_run {
                println!("{}", cfg.model.budget_table());
                println!("{}", serde_json::to_string_pretty(&cfg)?);
                println!("backend: {backend:?}");
                return Ok(());
            }

            tracing::info!(?backend, "starting training");
            match backend {
                BackendChoice::Ndarray => train_ndarray(cfg)?,
                BackendChoice::Wgpu => train_wgpu(cfg)?,
                BackendChoice::Cuda => train_cuda(cfg)?,
                BackendChoice::Vulkan => train_vulkan(cfg)?,
                BackendChoice::Rocm => train_rocm(cfg)?,
            }
        }
        Command::Eval {
            artifact_dir,
            config,
            model,
            tokenizer,
            ppl,
            blimp,
            generate,
            seq_len,
            stride,
            batch_size,
            temperature,
            seed,
            backend,
        } => {
            let run = EvalRun {
                artifact_dir,
                config_path: config,
                model_path: model,
                tokenizer_path: tokenizer,
                shard: ppl,
                blimp_dir: blimp,
                generate,
                corpus: EvalConfig {
                    seq_len,
                    // Half the window by default: every token then sees at least
                    // seq_len/2 tokens of context, at twice the compute. See
                    // `EvalConfig::stride`.
                    stride: stride.unwrap_or((seq_len / 2).max(1)),
                    batch_size,
                    num_workers: 2,
                },
                generation: GenerationConfig {
                    temperature,
                    seed,
                    ..Default::default()
                },
                blimp_batch_size: batch_size.max(2),
            };
            run.corpus.validate()?;

            let backend = backend.unwrap_or_default();
            tracing::info!(?backend, "starting evaluation");
            let report = match backend {
                BackendChoice::Ndarray => eval_ndarray(&run)?,
                BackendChoice::Wgpu => eval_wgpu(&run)?,
                BackendChoice::Cuda => eval_cuda(&run)?,
                BackendChoice::Vulkan => eval_vulkan(&run)?,
                BackendChoice::Rocm => eval_rocm(&run)?,
            };
            // stdout, not tracing: this is the artifact, and it should survive
            // being piped to a file without the log decoration.
            println!("{report}");
        }
    }
    Ok(())
}
