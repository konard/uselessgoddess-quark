//! Configuration and the analytic parameter budget for the compressor.
//!
//! Same discipline as [`ModelConfig`](crate::config::ModelConfig): every
//! parameter is countable from the config alone, the count is checked against
//! the constructed module in tests, and the rate is countable too. The issue
//! asks for code that is *logically verifiable* without training, and a budget
//! plus a bit-rate that can both be derived on paper is most of what that
//! means.

use serde::{Deserialize, Serialize};

use crate::{compress::quantize::Fsq, config::ModelConfig};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressConfig {
    /// Shape of *both* stacks: the encoder and the decoder are built from the
    /// same [`ModelConfig`], so each gets `n_layer_applications()` blocks.
    ///
    /// One config rather than two because the two stacks genuinely want the
    /// same width -- they exchange latents, and a width mismatch would buy
    /// nothing but a projection. Their *weights* are separate, which is the
    /// distinction that matters: compressing and expanding are different
    /// functions and get different parameters.
    pub model: ModelConfig,
    /// `N`: tokens per compressed span.
    pub span_len: usize,
    /// `K`: latent slots per span.
    pub n_slots: usize,
    /// Levels of the [`Fsq`] bottleneck, one entry per latent dimension.
    pub fsq_levels: Vec<u32>,
    /// Probability of replacing a decoder input token with the pad/BOS id.
    ///
    /// The single most important regularizer here, and the one the reference
    /// implementations in the issue omit. A teacher-forced autoregressive
    /// decoder can reconstruct most of a span from its own prefix without ever
    /// consulting the latent; corrupting the prefix is what forces information
    /// through the bottleneck rather than around it. CALM
    /// ([2510.27688](https://arxiv.org/abs/2510.27688)) uses 0.15 and measures
    /// the downstream metric moving 3.99 -> 4.70 when its regularization stack
    /// is added; DAAE ([1905.12777](https://arxiv.org/abs/1905.12777)) makes
    /// the same argument for latent-space smoothness.
    pub token_dropout: f64,
    /// Dropout applied to the quantized latent before the decoder reads it.
    ///
    /// CALM's second regularizer, same value and same purpose: a latent that
    /// survives perturbation is one a downstream LM body has some chance of
    /// *predicting in*. Without it "a small perturbation in the vector could
    /// decode into totally unrelated text" -- which would make the modular-LM
    /// end goal in issue #12 unreachable no matter how good reconstruction got.
    pub latent_dropout: f64,
    /// Token id the decoder starts from.
    ///
    /// Following `eval/blimp.rs`, which already uses the end-of-text token as a
    /// sequence-start marker: the tokenizer has no dedicated BOS, and reusing
    /// EOS is both the GPT-2 convention and the convention already established
    /// in this crate. The training path overwrites this with
    /// [`ShardMeta::eos_id`](crate::data::shard::ShardMeta), so the default
    /// here only matters for a hand-built config.
    #[serde(default)]
    pub bos_id: u32,
}

impl Default for CompressConfig {
    fn default() -> Self {
        Self::compressor_15m()
    }
}

impl CompressConfig {
    /// The reference compressor: ~15.0M parameters, 256 tokens -> 64 slots.
    ///
    /// **Why this shape.** The base-model question in issue #12 is whether to
    /// build on `quark_3m` (1 layer x 12 loops) or `quark_22m` (12 layers x 1
    /// loop). The answer is the untied one, and `docs/ANALYSIS.md` §0 has the
    /// measurement: untying is `+0 FLOPs` and `+0.30 GB VRAM` and moved
    /// WikiText-103 word perplexity from 108 to 74.965. Cross-layer sharing
    /// buys *storage*, and storage was never the binding constraint here -- the
    /// checkpoint is ~11 MB. For an autoencoder the argument is stronger still:
    /// reconstruction is capacity-bound rather than compute-bound, and the
    /// encoder's function (summarize) and the decoder's (expand) are different
    /// enough that forcing them through one shared weight set is exactly
    /// backwards.
    ///
    /// So: `n_loops = 1`, and four unique layers per stack. Eight layers total
    /// at `quark_3m`'s width lands at ~15.0M, which is deliberately inside the
    /// 13-17M range issue #12 reports for the model this is being compared
    /// against. A like-for-like budget is worth more than a bigger number.
    ///
    /// **Why 256 -> 64.** Four-fold token compression is the ratio ICAE
    /// ([2307.06945](https://arxiv.org/abs/2307.06945)) reports working well
    /// (BLEU 99.1 at 4x) and near where it reports 16x becoming
    /// "unsatisfactory" -- and ICAE had a 7B decoder. Capacity scales with the
    /// *decoder*, not the encoder ([2502.13063](https://arxiv.org/abs/2502.13063)
    /// measures 1568 tokens/vector at 8B against 96 at 410M), so at 15M the
    /// conservative end of the published range is the honest starting point.
    /// [`Self::rate_bits_per_token`] says what that costs in bits, which is the
    /// number that actually matters.
    pub fn compressor_15m() -> Self {
        Self {
            model: ModelConfig {
                n_unique_layers: 4,
                n_loops: 1,
                ..ModelConfig::quark_3m()
            },
            span_len: 256,
            n_slots: 64,
            fsq_levels: Fsq::default_levels().levels().to_vec(),
            token_dropout: 0.15,
            latent_dropout: 0.15,
            bos_id: 0,
        }
    }

    /// A model small enough to build in a unit test, with the same structure.
    pub fn tiny() -> Self {
        Self {
            model: ModelConfig {
                n_unique_layers: 2,
                n_loops: 1,
                max_seq_len: 64,
                ..ModelConfig::tiny()
            },
            span_len: 16,
            n_slots: 4,
            fsq_levels: vec![4, 3],
            token_dropout: 0.0,
            latent_dropout: 0.0,
            bos_id: 0,
        }
    }

    pub fn fsq(&self) -> Fsq {
        Fsq::new(self.fsq_levels.clone())
    }

    /// Tokens per slot: the *sequence-length* compression ratio, and the number
    /// the literature usually quotes as "the" compression ratio.
    ///
    /// Kept separate from [`Self::rate_bits_per_token`] and named for what it
    /// is, because it measures a saving in attention cost and KV-cache size,
    /// not in information. See the module docs of
    /// [`quantize`](crate::compress::quantize) for why conflating the two makes
    /// most published ratios unfalsifiable.
    pub fn token_ratio(&self) -> f64 {
        self.span_len as f64 / self.n_slots as f64
    }

    /// Bits the latent actually occupies, per source token.
    ///
    /// `K * sum(log2 L_i) / N`. This is the honest rate, and with a discrete
    /// bottleneck it is exact rather than an estimate. Against
    /// `log2(vocab) = 13` bits for a raw token id, the default config is
    /// `64 * 12.0948 / 256 = 3.024` bits/token -- a 4.30x reduction in bits,
    /// close to but not equal to the 4x token ratio, because a slot is a little
    /// narrower than a token.
    pub fn rate_bits_per_token(&self) -> f64 {
        self.n_slots as f64 * self.fsq().bits_per_slot() / self.span_len as f64
    }

    /// Total bits in one compressed span.
    pub fn span_bits(&self) -> usize {
        self.fsq().pack_bits(self.n_slots)
    }

    /// Whether the configured rate leaves room for *lossless* reconstruction of
    /// a source with the given per-token entropy.
    ///
    /// Shannon, applied honestly: a channel of `rate_bits_per_token()` cannot
    /// losslessly carry a source of higher entropy, whatever the architecture.
    /// This is not a validation error, because lossy is the intended regime --
    /// it is a diagnostic that says which side of the line a config sits on, so
    /// that a disappointing reconstruction number can be attributed to the
    /// *rate* rather than blamed on training.
    ///
    /// For calibration on WikiText-103 at vocab 8192: a raw token id is 13.0
    /// bits, and `quark_22m`'s measured word perplexity of 74.965 works out to
    /// roughly 4.6 bits/token. So the default 3.02 bits/token is below the
    /// source entropy and the compressor is lossy *by construction* -- which is
    /// exactly why the headline metric is free-running reconstruction accuracy
    /// and not a losslessness claim.
    pub fn is_lossless_feasible(&self, source_bits_per_token: f64) -> bool {
        self.rate_bits_per_token() >= source_bits_per_token
    }

    /// Longest sequence either stack has to attend over.
    ///
    /// The encoder reads `N + K` positions (span then slot queries), the
    /// decoder reads `K + N` (latents then the shifted span). Same number, and
    /// both must fit the RoPE table.
    pub fn max_positions(&self) -> usize {
        self.span_len + self.n_slots
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errs = Vec::new();
        if let Err(e) = self.model.validate() {
            errs.extend(e.into_iter().map(|s| format!("model: {s}")));
        }
        if let Err(e) = Fsq::try_new(self.fsq_levels.clone()) {
            errs.extend(e.into_iter().map(|s| format!("fsq: {s}")));
        }
        if self.span_len == 0 {
            errs.push("span_len must be >= 1".to_string());
        }
        if self.n_slots == 0 {
            errs.push("n_slots must be >= 1".to_string());
        }
        if self.n_slots >= self.span_len && self.span_len > 0 {
            errs.push(format!(
                "n_slots ({}) >= span_len ({}): that is not a compressor",
                self.n_slots, self.span_len
            ));
        }
        if self.max_positions() > self.model.max_seq_len {
            errs.push(format!(
                "span_len + n_slots = {} exceeds model.max_seq_len = {}; \
                 RoPE has no table for those positions",
                self.max_positions(),
                self.model.max_seq_len
            ));
        }
        for (name, p) in [
            ("token_dropout", self.token_dropout),
            ("latent_dropout", self.latent_dropout),
        ] {
            if !(0.0..1.0).contains(&p) {
                errs.push(format!("{name} must be in [0, 1), got {p}"));
            }
        }
        if self.bos_id as usize >= self.model.vocab_size {
            errs.push(format!(
                "bos_id ({}) is not a token in a vocabulary of {}",
                self.bos_id, self.model.vocab_size
            ));
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }

    /// The analytic parameter budget, itemized.
    ///
    /// Checked against `Module::num_params()` in the tests, so this table is a
    /// claim the compiler helps keep true rather than a comment that rots.
    pub fn budget(&self) -> Vec<BudgetEntry> {
        let m = &self.model;
        let per_layer = m
            .budget()
            .iter()
            .find(|e| e.name == "layers")
            .expect("ModelConfig::budget always reports a layers entry")
            .params
            / m.n_unique_layers;
        let d = self.fsq().dim();
        vec![
            // Shared between the two stacks, and tied to the output head: the
            // encoder and decoder must agree on what a token *is*, and the
            // reference implementation in issue #12 giving them separate tables
            // spends ~4M of a 13-17M budget saying the same thing twice.
            BudgetEntry {
                name: "token_embedding",
                params: m.vocab_size * m.d_emb,
            },
            BudgetEntry {
                name: "encoder_embed_proj",
                params: m.d_emb * m.d_model,
            },
            BudgetEntry {
                name: "decoder_embed_proj",
                params: m.d_emb * m.d_model,
            },
            BudgetEntry {
                name: "slot_queries",
                params: self.n_slots * m.d_model,
            },
            BudgetEntry {
                name: "encoder_layers",
                params: m.n_layer_applications() * per_layer,
            },
            BudgetEntry {
                name: "encoder_norm",
                params: m.d_model,
            },
            BudgetEntry {
                name: "to_latent",
                params: m.d_model * d,
            },
            BudgetEntry {
                name: "from_latent",
                params: d * m.d_model,
            },
            BudgetEntry {
                name: "decoder_layers",
                params: m.n_layer_applications() * per_layer,
            },
            BudgetEntry {
                name: "decoder_norm",
                params: m.d_model,
            },
            BudgetEntry {
                name: "unembed_proj",
                params: m.d_model * m.d_emb,
            },
        ]
    }

    pub fn param_count(&self) -> usize {
        self.budget().iter().map(|e| e.params).sum()
    }

    pub fn budget_table(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(s, "{:<24} {:>12}", "component", "params");
        let _ = writeln!(s, "{}", "-".repeat(37));
        for e in self.budget() {
            let _ = writeln!(s, "{:<24} {:>12}", e.name, e.params);
        }
        let _ = writeln!(s, "{}", "-".repeat(37));
        let _ = writeln!(s, "{:<24} {:>12}", "TOTAL", self.param_count());
        let _ = writeln!(
            s,
            "{:<24} {:>12.2}   (sequence length, not information)",
            "token ratio",
            self.token_ratio()
        );
        let _ = writeln!(
            s,
            "{:<24} {:>12.3}   (vs {:.1} for a raw token id)",
            "bits/token",
            self.rate_bits_per_token(),
            (self.model.vocab_size as f64).log2()
        );
        s
    }
}

/// One line of the compressor's parameter budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetEntry {
    pub name: &'static str,
    pub params: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reference_config_is_valid_and_in_the_intended_size_class() {
        let cfg = CompressConfig::compressor_15m();
        cfg.validate().unwrap();
        let n = cfg.param_count();
        // Issue #12 reports the model being compared against at 13-17M. Being
        // in that range is the point of the config, so it is a test.
        assert!(
            (13_000_000..=17_000_000).contains(&n),
            "{n} parameters is outside the 13-17M comparison class"
        );
    }

    #[test]
    fn tiny_is_valid_too() {
        CompressConfig::tiny().validate().unwrap();
    }

    /// The two ratios are different quantities and the config must not blur
    /// them. 4x on tokens is 4.30x on bits here, and the difference is the
    /// point: the test pins both so a change to `fsq_levels` cannot move the
    /// bit rate while the token ratio stays reassuringly put.
    #[test]
    fn token_ratio_and_bit_rate_are_reported_separately() {
        let cfg = CompressConfig::compressor_15m();
        assert!((cfg.token_ratio() - 4.0).abs() < 1e-12);
        assert!(
            (cfg.rate_bits_per_token() - 3.0237).abs() < 1e-3,
            "{}",
            cfg.rate_bits_per_token()
        );
        assert_eq!(cfg.span_bits(), 775);

        // And the honest consequence: below WikiText-103's per-token entropy,
        // so lossless is off the table and the docs say so.
        assert!(!cfg.is_lossless_feasible(4.6));
        assert!(cfg.is_lossless_feasible(3.0));
    }

    /// A config asking for more slots than tokens is an expander wearing a
    /// compressor's name, and no amount of training fixes that.
    #[test]
    fn a_config_that_does_not_compress_is_rejected() {
        let cfg = CompressConfig {
            n_slots: 256,
            ..CompressConfig::compressor_15m()
        };
        let errs = cfg.validate().unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("not a compressor")),
            "{errs:?}"
        );
    }

    /// RoPE is built for `max_seq_len` positions; asking either stack to run
    /// past it would fail at runtime, deep inside a training job, so it fails
    /// here instead.
    #[test]
    fn a_span_that_overruns_the_rope_table_is_rejected() {
        let cfg = CompressConfig {
            span_len: 1024,
            n_slots: 64,
            ..CompressConfig::compressor_15m()
        };
        let errs = cfg.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("max_seq_len")), "{errs:?}");
    }
}
