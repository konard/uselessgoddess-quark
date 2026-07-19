//! Evaluating a trained compressor: the metric `docs/COMPRESSION.md` §4 commits
//! to, and nothing that flatters it.
//!
//! Issue #12 built the compressor and its training run; there was no way to ask
//! a finished run how well it works, which is issue #14. The numbers here are
//! the ones §4 named, in the order it ranked them:
//!
//! * **H1, the headline** -- free-running greedy exact-match reconstruction
//!   accuracy. The decoder is fed *its own* output, so the only thing carrying
//!   the span across is the latent. This is the number to quote.
//! * **H1', the upper bound** -- teacher-forced accuracy, and the *gap* between
//!   the two. A large gap is the diagnostic that the decoder is leaning on its
//!   prefix rather than on the bottleneck, and it is invisible if only one of
//!   the two is reported.
//! * **H2, the rate** -- bits/token, exact rather than estimated because the
//!   bottleneck is discrete. §1.6: an accuracy without a rate beside it is
//!   unfalsifiable, so [`ReconstructionScore::report`] prints them together and
//!   there is no way to get one without the other.
//!
//! Reconstruction cross-entropy is reported last and labelled for what it is:
//! the training loss. §4 -- "reporting it as a result is grading your own
//! homework".
//!
//! **What this module deliberately does not do.** H3 (`CR@99`) is a curve over
//! several *trained* compressors, so it belongs to a sweep script and not to a
//! single run's evaluation. H4 (downstream retention) needs a second model -- a
//! `QuarkLm` to score the round-tripped text -- which is a different command
//! with a different pair of artifacts.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    prelude::Backend,
    tensor::{ElementConversion, Int, Tensor, TensorData},
};
use tokenizers::Tokenizer;

use crate::{
    compress::Compressor,
    data::{tokenizer, Shard, TokenBatcher, TokenDataset},
    train::output::masked_cross_entropy,
};

/// How to sweep a shard for reconstruction.
///
/// No `seq_len` and no `stride`: a compressor's window length is its
/// `span_len` -- it asserts the span it was given is exactly that -- and the
/// windows are disjoint, because a strided sweep would reconstruct the same
/// token from two different latents and count it twice.
#[derive(Debug, Clone)]
pub struct CompressEvalConfig {
    pub batch_size: usize,
    pub num_workers: usize,
    /// Stop after this many spans. `None` sweeps the whole shard.
    ///
    /// Free-running decoding is `span_len` sequential forward passes per span
    /// -- for the reference config, 256 of them -- so a full sweep of a
    /// WikiText-sized shard is hours, not minutes. A capped sweep is an honest
    /// measurement of a stated number of spans, and [`ReconstructionScore`]
    /// reports how many it covered so nobody has to guess.
    pub max_spans: Option<usize>,
}

impl Default for CompressEvalConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            num_workers: 2,
            // A cap by default, unlike the LM's corpus sweep, because the cost
            // per span is a whole decode rather than one forward pass. 512
            // spans at the reference span_len is 131k reconstructed tokens --
            // enough that the accuracy's standard error is well under a tenth
            // of a point, and small enough to run while you watch.
            max_spans: Some(512),
        }
    }
}

impl CompressEvalConfig {
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            bail!("batch_size must be positive");
        }
        if self.max_spans == Some(0) {
            bail!("max_spans must be positive, or None for the whole shard");
        }
        Ok(())
    }
}

/// What a reconstruction sweep measured.
///
/// Raw counts, following [`CorpusScore`](crate::eval::CorpusScore): every ratio
/// below is derived here, so no accumulator is carrying a number in a unit
/// someone happened to want at the time.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconstructionScore {
    pub n_spans: usize,
    /// Spans the shard could have offered, before `max_spans`.
    pub n_spans_available: usize,
    /// Tokens covered: `n_spans * span_len`.
    pub n_tokens: usize,
    /// Tokens the free-running decode got exactly right. **The headline.**
    pub free_running_correct: usize,
    /// Tokens the teacher-forced pass got right. An upper bound.
    pub teacher_forced_correct: usize,
    /// Spans reconstructed free-running with *every* token correct.
    pub exact_spans: usize,
    /// Teacher-forced negative log-likelihood in nats, summed over every token.
    pub total_nll: f64,
    /// `span_len / n_slots`, carried so the report is self-contained.
    pub token_ratio: f64,
    /// `K * sum(log2 L_i) / N`.
    pub bits_per_token: f64,
    /// `log2(vocab)`: what a raw token id costs, for the comparison.
    pub token_bits: f64,
}

impl ReconstructionScore {
    /// **H1.** The fraction of tokens the model recovers from the latent alone.
    pub fn free_running_accuracy(&self) -> f64 {
        self.ratio(self.free_running_correct)
    }

    /// The upper bound: same accuracy, with the true prefix handed back at
    /// every step. Not a result on its own -- see the module docs.
    pub fn teacher_forced_accuracy(&self) -> f64 {
        self.ratio(self.teacher_forced_correct)
    }

    /// How much of the teacher-forced accuracy came from the prefix rather than
    /// from the bottleneck. Large means the decoder is a language model wearing
    /// a compressor's name.
    pub fn exposure_gap(&self) -> f64 {
        self.teacher_forced_accuracy() - self.free_running_accuracy()
    }

    /// Spans recovered token-for-token. Harsher than the token accuracy by
    /// construction, and the one that answers "can I hand this latent to
    /// something and get my text back".
    pub fn exact_span_rate(&self) -> f64 {
        if self.n_spans == 0 {
            return 0.0;
        }
        self.exact_spans as f64 / self.n_spans as f64
    }

    /// Teacher-forced perplexity: the training loss, exponentiated. Reported
    /// for continuity with the training log, not as a result.
    pub fn token_ppl(&self) -> f64 {
        if self.n_tokens == 0 {
            return f64::INFINITY;
        }
        (self.total_nll / self.n_tokens as f64).exp()
    }

    /// Compression in *bits*, which is the ratio the rate actually buys.
    pub fn bit_ratio(&self) -> f64 {
        if self.bits_per_token == 0.0 {
            return f64::INFINITY;
        }
        self.token_bits / self.bits_per_token
    }

    fn ratio(&self, n: usize) -> f64 {
        if self.n_tokens == 0 {
            return 0.0;
        }
        n as f64 / self.n_tokens as f64
    }

    pub fn report(&self) -> String {
        format!(
            "free-running accuracy   {:>10.4}%   <- the headline (latent only)\n\
             exact-span rate         {:>10.4}%   (spans recovered token-for-token)\n\
             teacher-forced accuracy {:>10.4}%   (upper bound; the decoder saw the true prefix)\n\
             exposure gap            {:>10.4}%   (large => the decoder leans on its prefix)\n\
             bits per token          {:>10.3}    ({:.2}x vs {:.1} bits for a raw id)\n\
             token ratio             {:>10.2}    (sequence length, not information)\n\
             reconstruction NLL      {:>10.4}    (the training loss; ppl {:.3})\n\
             spans                   {:>10}    ({} available in the shard)\n\
             tokens                  {:>10}",
            self.free_running_accuracy() * 100.0,
            self.exact_span_rate() * 100.0,
            self.teacher_forced_accuracy() * 100.0,
            self.exposure_gap() * 100.0,
            self.bits_per_token,
            self.bit_ratio(),
            self.token_bits,
            self.token_ratio,
            self.total_nll / self.n_tokens.max(1) as f64,
            self.token_ppl(),
            self.n_spans,
            self.n_spans_available,
            self.n_tokens,
        )
    }
}

/// Sweep `shard` and measure how much of it survives the bottleneck.
///
/// Both passes run on the same batch, so the teacher-forced number and the
/// free-running number describe the same tokens. Measuring them on different
/// samples would make the gap -- the point of reporting both -- meaningless.
pub fn evaluate<B: Backend>(
    model: &Compressor<B>,
    shard: Arc<Shard>,
    config: &CompressEvalConfig,
    device: &B::Device,
) -> Result<ReconstructionScore> {
    config.validate()?;

    let cfg = model.config().clone();
    let meta = shard.meta().clone();
    if meta.vocab_size != cfg.model.vocab_size {
        bail!(
            "shard vocab {} != model vocab {}: this shard was tokenized with a \
             different tokenizer, and its ids mean nothing to this model",
            meta.vocab_size,
            cfg.model.vocab_size
        );
    }

    let dataset = TokenDataset::train(shard, cfg.span_len);
    let n_spans_available = dataset.len();
    if n_spans_available == 0 {
        bail!(
            "shard holds {} tokens, too few for even one {}-token span",
            meta.n_tokens,
            cfg.span_len
        );
    }

    let loader = DataLoaderBuilder::new(TokenBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(dataset);

    let budget = config.max_spans.unwrap_or(usize::MAX);
    let mut score = ReconstructionScore {
        n_spans: 0,
        n_spans_available,
        n_tokens: 0,
        free_running_correct: 0,
        teacher_forced_correct: 0,
        exact_spans: 0,
        total_nll: 0.0,
        token_ratio: cfg.token_ratio(),
        bits_per_token: cfg.rate_bits_per_token(),
        token_bits: (cfg.model.vocab_size as f64).log2(),
    };

    for batch in loader.iter() {
        if score.n_spans >= budget {
            break;
        }
        // The last batch may overshoot the budget; trim it rather than
        // reporting more spans than were asked for.
        let take = (budget - score.n_spans).min(batch.input.dims()[0]);
        let span = batch.input.slice([0..take, 0..cfg.span_len]);

        // Teacher-forced, in the exact form the training step used: the span is
        // its own target, and every position is scored.
        let ones = Tensor::ones([take, cfg.span_len], device);
        let logits = model.forward(span.clone());
        let argmax = logits.clone().argmax(2).reshape([take, cfg.span_len]);
        score.teacher_forced_correct += n_equal(argmax, span.clone());

        let out = masked_cross_entropy(logits, span.clone(), ones);
        score.total_nll += out.sum_nll.into_scalar().elem::<f64>();

        // Free-running: the decoder sees its own output and the latent, and
        // nothing else.
        let recon = model.reconstruct(span.clone());
        let matches = recon.equal(span).int().sum_dim(1).reshape([take]);
        score.free_running_correct += matches.clone().sum().into_scalar().elem::<f64>() as usize;
        score.exact_spans += n_equal(
            matches,
            Tensor::<B, 1, Int>::full([take], cfg.span_len as i64, device),
        );

        score.n_spans += take;
        score.n_tokens += take * cfg.span_len;
    }

    Ok(score)
}

/// How many positions of two identically-shaped id tensors agree.
fn n_equal<B: Backend, const D: usize>(a: Tensor<B, D, Int>, b: Tensor<B, D, Int>) -> usize {
    a.equal(b).int().sum().into_scalar().elem::<f64>() as usize
}

/// One span, before and after the round trip.
///
/// A scalar accuracy cannot distinguish "recovered the sentence with two
/// articles wrong" from "emitted fluent text about something else", and both
/// are obvious to a human reading four samples. Same argument as
/// [`crate::eval::generate`], which exists for the language model's version of
/// this problem.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconstructionSample {
    pub original: String,
    pub reconstruction: String,
    /// Tokens correct in this span, out of `span_len`.
    pub correct: usize,
    pub span_len: usize,
}

impl ReconstructionSample {
    pub fn accuracy(&self) -> f64 {
        self.correct as f64 / self.span_len.max(1) as f64
    }
}

/// Round-trip the first `n` spans of `shard` and decode both sides to text.
pub fn samples<B: Backend>(
    model: &Compressor<B>,
    tok: &Tokenizer,
    shard: Arc<Shard>,
    n: usize,
    device: &B::Device,
) -> Result<Vec<ReconstructionSample>> {
    let cfg = model.config().clone();
    let dataset = TokenDataset::train(shard, cfg.span_len);
    let n = n.min(dataset.len());
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let window = dataset.get(i).expect("index below len");
        let ids: Vec<i32> = window.input.iter().map(|&t| t as i32).collect();
        let span = Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [1, cfg.span_len]), device);

        let recon = model.reconstruct(span.clone());
        let correct = n_equal(recon.clone(), span);
        let recon: Vec<u32> = recon
            .into_data()
            .to_vec::<i64>()
            .expect("reconstruct returns integer ids")
            .into_iter()
            .map(|t| t as u32)
            .collect();

        out.push(ReconstructionSample {
            original: tokenizer::decode(tok, &window.input)?,
            reconstruction: tokenizer::decode(tok, &recon)?,
            correct,
            span_len: cfg.span_len,
        });
    }
    Ok(out)
}

/// Render samples for a human, with the separator stripped the way
/// [`crate::eval::generate::report`] strips it.
pub fn report(samples: &[ReconstructionSample]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for (i, sample) in samples.iter().enumerate() {
        let _ = writeln!(
            s,
            "--- span {i} ({}/{} tokens, {:.1}%) ---\n  in : {}\n  out: {}\n",
            sample.correct,
            sample.span_len,
            sample.accuracy() * 100.0,
            clean(&sample.original),
            clean(&sample.reconstruction),
        );
    }
    s
}

/// The document separator is an artifact of our own shard layout; a reader
/// should never meet it in a sample.
fn clean(text: &str) -> String {
    text.replace(tokenizer::EOS_TOKEN, " ")
}

/// Rebuild a trained compressor from a run's artifact directory.
///
/// The parameter-count check is the one from [`crate::eval::load_model`], and it
/// is here for the same measured reason: burn's recorder installs a record's
/// shapes without consulting the module it is loading into, so a `config.json`
/// that does not belong to these weights produces a model that is neither, and
/// reports an accuracy rather than an error.
pub fn load_compressor<B: Backend>(
    config: &crate::compress::CompressTrainConfig,
    model_path: &std::path::Path,
    device: &B::Device,
) -> Result<Compressor<B>> {
    use burn::{
        module::Module,
        record::{CompactRecorder, FileRecorder},
    };

    let extension = <CompactRecorder as FileRecorder<B>>::file_extension();
    let record = model_path.with_extension(extension);
    if !record.is_file() {
        bail!(
            "{}",
            crate::eval::missing_record_message(model_path, &record, extension)
        );
    }

    let model = Compressor::<B>::new(config.compress.clone(), device)
        .load_file(model_path, &CompactRecorder::new(), device)
        .with_context(|| format!("loading weights from {}", model_path.display()))?;

    let loaded = model.num_params();
    let expected = config.compress.param_count();
    if loaded != expected {
        bail!(
            "the record at {} holds {loaded} parameters but the config describes a \
             {expected}-parameter compressor: the config does not belong to these weights",
            model_path.display(),
        );
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compress::{CompressConfig, CompressTrainConfig},
        data::ShardWriter,
        test_util::TestBackend,
        TrainConfig,
    };
    use burn::{module::Module, record::CompactRecorder};
    use std::path::Path;

    fn shard(dir: &Path, n_tokens: usize, vocab: usize) -> Arc<Shard> {
        let bin = dir.join("c.bin");
        let mut w = ShardWriter::create(&bin, vocab, 0).unwrap();
        let tokens: Vec<u32> = (0..n_tokens as u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 16) % vocab as u32)
            .collect();
        w.push_document("one two three four five", &tokens).unwrap();
        w.finish().unwrap();
        Arc::new(Shard::open(&bin).unwrap())
    }

    fn cfg() -> CompressEvalConfig {
        CompressEvalConfig {
            batch_size: 2,
            num_workers: 1,
            max_spans: None,
        }
    }

    /// The sweep must run end to end and produce counts that are internally
    /// consistent -- every ratio a proportion of the tokens it claims to cover.
    #[test]
    fn a_sweep_reports_counts_that_add_up() {
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();
        let c = CompressConfig::tiny();
        let model = Compressor::<TestBackend>::new(c.clone(), &device);
        let s = shard(dir.path(), 200, c.model.vocab_size);

        let score = evaluate(&model, s, &cfg(), &device).unwrap();

        assert_eq!(score.n_tokens, score.n_spans * c.span_len);
        assert!(score.n_spans > 0);
        assert!(score.free_running_correct <= score.n_tokens);
        assert!(score.teacher_forced_correct <= score.n_tokens);
        assert!(score.exact_spans <= score.n_spans);
        assert!((0.0..=1.0).contains(&score.free_running_accuracy()));
        // The rate is the config's, not something the sweep invented.
        assert!((score.bits_per_token - c.rate_bits_per_token()).abs() < 1e-12);
        assert!((score.token_ratio - c.token_ratio()).abs() < 1e-12);
        // An untrained model reconstructs nothing, so its loss sits at ln(V).
        let per_token = score.total_nll / score.n_tokens as f64;
        let uniform = (c.model.vocab_size as f64).ln();
        assert!(
            (per_token - uniform).abs() < 0.5,
            "per-token NLL {per_token} should sit near ln(V) = {uniform}"
        );
    }

    /// `max_spans` is what makes a free-running sweep affordable, so it has to
    /// bound the work *exactly* -- a cap that overshoots by a batch would make
    /// two capped runs on different batch sizes incomparable.
    #[test]
    fn max_spans_bounds_the_sweep_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();
        let c = CompressConfig::tiny();
        let model = Compressor::<TestBackend>::new(c.clone(), &device);
        let s = shard(dir.path(), 200, c.model.vocab_size);

        let full = evaluate(&model, s.clone(), &cfg(), &device).unwrap();
        assert!(full.n_spans > 3, "the shard should offer several spans");

        let capped = evaluate(
            &model,
            s,
            &CompressEvalConfig {
                // Deliberately not a multiple of `batch_size`.
                max_spans: Some(3),
                ..cfg()
            },
            &device,
        )
        .unwrap();
        assert_eq!(capped.n_spans, 3);
        assert_eq!(capped.n_tokens, 3 * c.span_len);
        // And it still says how much it left on the table.
        assert_eq!(capped.n_spans_available, full.n_spans_available);
    }

    /// The teacher-forced pass is an upper bound on the free-running one, and
    /// the report says so. If the inequality ever inverted, the two passes
    /// would not be scoring the same tokens and the gap would be noise.
    #[test]
    fn teacher_forcing_is_an_upper_bound() {
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();
        // Regularizers off: they are training-only, but a sweep that measured
        // them would make this comparison depend on the RNG.
        let c = CompressConfig {
            token_dropout: 0.0,
            latent_dropout: 0.0,
            ..CompressConfig::tiny()
        };
        let model = Compressor::<TestBackend>::new(c.clone(), &device);
        let s = shard(dir.path(), 200, c.model.vocab_size);

        let score = evaluate(&model, s, &cfg(), &device).unwrap();
        assert!(
            score.teacher_forced_correct >= score.free_running_correct,
            "teacher forcing scored {} against free-running {}",
            score.teacher_forced_correct,
            score.free_running_correct
        );
        assert!(score.exposure_gap() >= 0.0);
    }

    /// The same guard the LM evaluator has: ids from another tokenizer would
    /// produce a plausible-looking accuracy rather than an error.
    #[test]
    fn a_shard_from_another_tokenizer_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();
        let c = CompressConfig::tiny();
        let model = Compressor::<TestBackend>::new(c, &device);
        let s = shard(dir.path(), 200, 512);

        let err = evaluate(&model, s, &cfg(), &device)
            .unwrap_err()
            .to_string();
        assert!(err.contains("different tokenizer"), "got: {err}");
    }

    /// A shard shorter than one span has no reconstruction to report, and
    /// should say which of the two numbers is the problem.
    #[test]
    fn a_shard_too_short_for_one_span_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();
        let c = CompressConfig::tiny();
        let model = Compressor::<TestBackend>::new(c.clone(), &device);
        let s = shard(dir.path(), c.span_len / 2, c.model.vocab_size);

        let err = evaluate(&model, s, &cfg(), &device)
            .unwrap_err()
            .to_string();
        assert!(err.contains("too few"), "got: {err}");
    }

    /// A saved compressor must come back with the same weights, and a config
    /// that describes a *different* compressor must be refused rather than
    /// silently loaded -- burn's recorder does not check shapes.
    #[test]
    fn a_config_that_does_not_belong_to_the_weights_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();
        let c = CompressConfig::tiny();
        let model_path = dir.path().join("model");
        Compressor::<TestBackend>::new(c.clone(), &device)
            .save_file(&model_path, &CompactRecorder::new())
            .unwrap();

        let good = CompressTrainConfig::sync(c.clone(), TrainConfig::default());
        load_compressor::<TestBackend>(&good, &model_path, &device).unwrap();

        let wrong = CompressTrainConfig::sync(
            CompressConfig {
                n_slots: c.n_slots * 2,
                ..c
            },
            TrainConfig::default(),
        );
        let err = load_compressor::<TestBackend>(&wrong, &model_path, &device)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("does not belong to these weights"),
            "got: {err}"
        );
    }

    /// The record that is not there must be named, with the neighbours that
    /// are -- the same failure that cost four guesses in PR #11.
    #[test]
    fn a_missing_record_names_the_file_it_looked_for() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = CompressTrainConfig::sync(CompressConfig::tiny(), TrainConfig::default());
        let err =
            load_compressor::<TestBackend>(&cfg, &dir.path().join("mdoel"), &Default::default())
                .unwrap_err()
                .to_string();
        assert!(err.contains("mdoel.mpk"), "got: {err}");
    }
}
