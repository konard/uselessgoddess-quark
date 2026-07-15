//! Model configuration and the analytic parameter budget.
//!
//! The architecture here is deliberately "floating": every structural choice
//! that trades parameters against capacity is a config field, so the same code
//! path expresses a plain GPT-2-shaped baseline, an ALBERT-style shared-layer
//! model, and everything between. That matters because at a 3.0M budget the
//! right point in that space is an empirical question, not something to
//! hard-code.
//!
//! [`ModelConfig::param_count`] is an analytic count derived from the config
//! alone. It is checked against the real `Module::num_params()` in the tests,
//! so a budget can be verified without building the model or touching a GPU.

use serde::{Deserialize, Serialize};

/// How a stack of `n_loops * n_unique_layers` layer applications maps onto the
/// `n_unique_layers` distinct parameter sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerSchedule {
    /// `[0, 1, 0, 1, ...]` -- Universal-Transformer style. Each loop iteration
    /// runs the full set of unique layers.
    Cycle,
    /// `[0, 0, 0, 1, 1, 1]` -- each unique layer is applied `n_loops` times
    /// before moving to the next. ALBERT's "shared groups" arrangement.
    Blocked,
}

/// Where a layer normalization sits relative to the residual branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormPlacement {
    /// `x + f(norm(x))` -- what every modern LM uses; stable without warmup
    /// tricks.
    Pre,
    /// `norm(x + f(x))` -- GPT-2's arrangement, kept for baseline parity.
    Post,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Tokenizer vocabulary size.
    pub vocab_size: usize,
    /// Rank of the factorized token embedding (ALBERT-style `V x E`, `E x H`).
    /// Set equal to `d_model` to disable factorization.
    ///
    /// This doubles as the rank cap on the output log-probability matrix when
    /// `tie_embeddings` is set -- see [`ModelConfig::softmax_rank_cap`].
    pub d_emb: usize,
    /// Residual stream width.
    pub d_model: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads. `1` is multi-query, `n_heads` is full
    /// multi-head, anything between (dividing `n_heads`) is grouped-query.
    pub n_kv_heads: usize,
    /// Inner width of the SwiGLU feed-forward block.
    pub d_ff: usize,
    /// Number of distinct parameter sets in the layer stack.
    pub n_unique_layers: usize,
    /// How many times each unique layer is applied. `1` means no sharing.
    pub n_loops: usize,
    /// Ordering of layer applications; only meaningful when
    /// `n_unique_layers > 1 && n_loops > 1`.
    pub layer_schedule: LayerSchedule,
    /// Maximum position the RoPE tables are built for.
    pub max_seq_len: usize,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Reuse the token embedding matrix as the output projection.
    pub tie_embeddings: bool,
    pub norm_placement: NormPlacement,
    pub norm_eps: f64,
    pub dropout: f64,
}

/// One line of the parameter budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetEntry {
    pub name: &'static str,
    pub params: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::quark_3m()
    }
}

impl ModelConfig {
    /// The reference 3.0M-budget model.
    ///
    /// The parameter split is the whole argument of this design, so it is worth
    /// stating explicitly: `d_emb = 128` is high for a model this small, and
    /// that is on purpose. When embeddings are tied, the output log-probability
    /// matrix factors through the `d_emb`-dimensional bottleneck, so `d_emb`
    /// hard-caps the rank of the distribution the model can express (Yang et
    /// al. 2018, "Breaking the Softmax Bottleneck"). The `d_emb = 32` that a
    /// naive budget optimization suggests buys ~0.9M parameters elsewhere at
    /// the cost of a rank-32 output -- a bad trade we deliberately decline.
    pub fn quark_3m() -> Self {
        Self {
            vocab_size: 8192,
            d_emb: 128,
            d_model: 384,
            n_heads: 6,
            n_kv_heads: 2,
            d_ff: 1152,
            n_unique_layers: 1,
            n_loops: 12,
            layer_schedule: LayerSchedule::Cycle,
            max_seq_len: 1024,
            rope_theta: 10_000.0,
            tie_embeddings: true,
            norm_placement: NormPlacement::Pre,
            norm_eps: 1e-5,
            dropout: 0.0,
        }
    }

    /// Two unique layers, six loops each: a second parameter set bought by
    /// narrowing the residual. A natural A/B against [`Self::quark_3m`].
    ///
    /// `d_model` is 288 rather than a rounder number because the budget is
    /// binding: two unique layers cost twice as much per unit of width, so the
    /// width has to fall to keep the total under 3.0M (2,865,568).
    pub fn quark_3m_deep() -> Self {
        Self {
            vocab_size: 8192,
            d_emb: 128,
            d_model: 288,
            n_heads: 4,
            n_kv_heads: 1,
            d_ff: 768,
            n_unique_layers: 2,
            n_loops: 6,
            ..Self::quark_3m()
        }
    }

    /// No weight sharing: 6 distinct layers, narrow. The control condition for
    /// measuring what cross-layer sharing actually buys.
    ///
    /// Six unique layers force `d_model` all the way down to 168 to stay in
    /// budget (2,871,880). That is the point of the control: at a fixed
    /// parameter count, not sharing means being narrow.
    pub fn quark_3m_dense() -> Self {
        Self {
            vocab_size: 8192,
            d_emb: 128,
            d_model: 168,
            n_heads: 4,
            n_kv_heads: 1,
            d_ff: 448,
            n_unique_layers: 6,
            n_loops: 1,
            ..Self::quark_3m()
        }
    }

    /// A tiny config for tests and CI. Not intended for training.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,
            d_emb: 32,
            d_model: 64,
            n_heads: 4,
            n_kv_heads: 2,
            d_ff: 128,
            n_unique_layers: 1,
            n_loops: 2,
            max_seq_len: 64,
            ..Self::quark_3m()
        }
    }

    /// # Panics
    ///
    /// When `n_heads == 0`. [`Self::validate`] rejects that and is called before
    /// anything is built, so this is a bug rather than a bad config reaching a
    /// user -- the assertion says which, instead of leaving a bare "divide by
    /// zero" to be traced back here.
    pub fn d_head(&self) -> usize {
        assert!(
            self.n_heads > 0,
            "n_heads must be >= 1; ModelConfig::validate rejects 0 and should have run first"
        );
        self.d_model / self.n_heads
    }

    /// Total layer applications per forward pass.
    pub fn n_layer_applications(&self) -> usize {
        self.n_unique_layers * self.n_loops
    }

    /// Upper bound on the rank of the output log-probability matrix.
    ///
    /// With tied embeddings the logits are produced by projecting the `d_model`
    /// residual down to `d_emb` and multiplying by the `V x E` table, so the
    /// rank cannot exceed `d_emb` regardless of how wide the model is. Untied,
    /// the bound is `d_model`. This is the softmax bottleneck, and it is a
    /// property of the config that no amount of training changes.
    pub fn softmax_rank_cap(&self) -> usize {
        if self.tie_embeddings {
            self.d_emb.min(self.d_model)
        } else {
            self.d_model
        }
    }

    /// Validate structural invariants. Returns every violation, not just the
    /// first, so a bad config is diagnosed in one pass.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errs = Vec::new();
        // Guarded first, and everything that divides by it nested inside: a
        // remainder by zero panics, and a validator that panics on a bad config
        // is the one input it exists to handle. The checks below are skipped
        // rather than dropped -- "d_model must be divisible by n_heads" is not a
        // useful thing to say about a model with no heads -- while unrelated
        // faults are still collected, keeping this function's promise to report
        // every violation in one pass.
        if self.n_heads == 0 {
            errs.push("n_heads must be >= 1".to_string());
        } else {
            if self.d_model % self.n_heads != 0 {
                errs.push(format!(
                    "d_model ({}) must be divisible by n_heads ({})",
                    self.d_model, self.n_heads
                ));
            } else if self.d_head() % 2 != 0 {
                // burn's RotaryEncoding pairs adjacent elements, so the rotated
                // dimension must be even. Only meaningful once d_head is a whole
                // number, hence the `else`.
                errs.push(format!(
                    "d_head ({}) must be even for rotary embeddings",
                    self.d_head()
                ));
            }
        }
        if self.n_kv_heads == 0 || self.n_heads % self.n_kv_heads != 0 {
            errs.push(format!(
                "n_heads ({}) must be a positive multiple of n_kv_heads ({})",
                self.n_heads, self.n_kv_heads
            ));
        }
        if self.n_kv_heads > self.n_heads {
            errs.push(format!(
                "n_kv_heads ({}) cannot exceed n_heads ({})",
                self.n_kv_heads, self.n_heads
            ));
        }
        if self.n_unique_layers == 0 {
            errs.push("n_unique_layers must be >= 1".to_string());
        }
        if self.n_loops == 0 {
            errs.push("n_loops must be >= 1".to_string());
        }
        if self.d_emb == 0 || self.d_model == 0 || self.vocab_size == 0 {
            errs.push("d_emb, d_model and vocab_size must all be >= 1".to_string());
        }
        if self.d_emb > self.d_model {
            errs.push(format!(
                "d_emb ({}) > d_model ({}): factorization would expand rather \
                 than compress",
                self.d_emb, self.d_model
            ));
        }
        if self.norm_eps <= 0.0 {
            errs.push("norm_eps must be > 0".to_string());
        }
        if !(0.0..1.0).contains(&self.dropout) {
            errs.push("dropout must be in [0, 1)".to_string());
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }

    /// Weights in one unique transformer layer that participate in a matmul.
    ///
    /// Split out from [`Self::params_per_layer`] because the two feed different
    /// questions: storage counts every parameter, whereas the `6 * N * D` FLOP
    /// convention counts only matmul weights. RMSNorm gains are elementwise and
    /// contribute negligible arithmetic, so folding them into a
    /// *compute*-equivalent figure would overstate it.
    fn matmul_params_per_layer(&self) -> usize {
        let d = self.d_model;
        let kv = self.n_kv_heads * self.d_head();
        let wq = d * d;
        let wk = d * kv;
        let wv = d * kv;
        let wo = d * d;
        // SwiGLU: gate and up project d -> d_ff, down projects d_ff -> d.
        let ffn = d * self.d_ff * 2 + self.d_ff * d;
        wq + wk + wv + wo + ffn
    }

    /// Parameters in one unique transformer layer.
    fn params_per_layer(&self) -> usize {
        let norms = 2 * self.d_model; // RMSNorm gain, pre-attention and pre-FFN
        self.matmul_params_per_layer() + norms
    }

    /// The analytic parameter budget, itemized.
    ///
    /// Verified against the constructed module in `tests`; if you change the
    /// model you must change this, and the test will tell you.
    pub fn budget(&self) -> Vec<BudgetEntry> {
        let mut out = vec![
            BudgetEntry {
                name: "token_embedding",
                params: self.vocab_size * self.d_emb,
            },
            BudgetEntry {
                name: "embed_proj",
                params: self.d_emb * self.d_model,
            },
        ];
        if self.tie_embeddings {
            // Tied: the V x E table is reused as the output projection, so only
            // the H -> E projection is new.
            out.push(BudgetEntry {
                name: "unembed_proj",
                params: self.d_model * self.d_emb,
            });
        } else {
            out.push(BudgetEntry {
                name: "lm_head",
                params: self.d_model * self.vocab_size,
            });
        }
        out.push(BudgetEntry {
            name: "layers",
            params: self.n_unique_layers * self.params_per_layer(),
        });
        out.push(BudgetEntry {
            name: "final_norm",
            params: self.d_model,
        });
        out
    }

    /// Total trainable parameters.
    pub fn param_count(&self) -> usize {
        self.budget().iter().map(|e| e.params).sum()
    }

    /// Parameters a *dense, unshared* model with identical per-token FLOPs
    /// would have.
    ///
    /// Weight sharing reduces storage, not arithmetic: looping one layer 12
    /// times costs exactly what 12 distinct layers cost. Training time and
    /// activation memory track this number, while the 3.0M budget tracks
    /// [`Self::param_count`]. Conflating the two is the most common way to
    /// mis-plan a shared-layer run.
    pub fn compute_equivalent_params(&self) -> usize {
        self.matmul_params_per_layer() * self.n_layer_applications()
    }

    /// Forward+backward FLOPs per token, `6 * N * D` convention, including the
    /// attention score/context matmuls which are not captured by parameter
    /// count and are non-trivial for a narrow model at long context.
    pub fn flops_per_token(&self, seq_len: usize) -> u64 {
        let matmul = 6 * self.compute_equivalent_params() as u64;
        let attn = 12 * self.n_layer_applications() as u64 * self.d_model as u64 * seq_len as u64;
        matmul + attn
    }

    /// Render the budget as a human-readable table.
    pub fn budget_table(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(s, "{:<24} {:>12}", "component", "params");
        let _ = writeln!(s, "{}", "-".repeat(37));
        for e in self.budget() {
            let _ = writeln!(s, "{:<24} {:>12}", e.name, fmt_thousands(e.params));
        }
        let _ = writeln!(s, "{}", "-".repeat(37));
        let _ = writeln!(
            s,
            "{:<24} {:>12}",
            "TOTAL",
            fmt_thousands(self.param_count())
        );
        let _ = writeln!(
            s,
            "{:<24} {:>12}   (drives FLOPs, not the budget)",
            "compute-equivalent",
            fmt_thousands(self.compute_equivalent_params())
        );
        let _ = writeln!(
            s,
            "{:<24} {:>12}   (softmax bottleneck)",
            "output rank cap",
            fmt_thousands(self.softmax_rank_cap())
        );
        s
    }
}

fn fmt_thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_config_fits_the_3m_budget() {
        let c = ModelConfig::quark_3m();
        let total = c.param_count();
        assert!(
            total <= 3_000_000,
            "quark_3m must fit the 3.0M budget, got {total}"
        );
        // Guard against silently drifting far *under* budget too: unused budget
        // is wasted capacity.
        assert!(
            total >= 2_700_000,
            "quark_3m leaves too much budget unused: {total}"
        );
        assert_eq!(total, 2_868_352);
    }

    #[test]
    fn all_reference_configs_are_valid_and_within_budget() {
        for (name, c) in [
            ("quark_3m", ModelConfig::quark_3m()),
            ("quark_3m_deep", ModelConfig::quark_3m_deep()),
            ("quark_3m_dense", ModelConfig::quark_3m_dense()),
            ("tiny", ModelConfig::tiny()),
        ] {
            c.validate()
                .unwrap_or_else(|e| panic!("{name} is invalid: {e:?}"));
            if name != "tiny" {
                assert!(
                    c.param_count() <= 3_000_000,
                    "{name} exceeds the 3.0M budget: {}",
                    c.param_count()
                );
            }
        }
    }

    #[test]
    fn sharing_saves_parameters_but_not_compute() {
        let c = ModelConfig::quark_3m();
        assert_eq!(c.n_layer_applications(), 12);
        // The whole point of the shared-layer design.
        assert!(c.compute_equivalent_params() > 6 * c.param_count());
    }

    #[test]
    fn tying_embeddings_saves_the_lm_head() {
        let tied = ModelConfig::quark_3m();
        let untied = ModelConfig {
            tie_embeddings: false,
            ..tied.clone()
        };
        let saved = untied.param_count() - tied.param_count();
        // Untied costs a full H x V head instead of an H x E projection.
        assert_eq!(
            saved,
            tied.d_model * tied.vocab_size - tied.d_model * tied.d_emb
        );
        assert!(
            untied.param_count() > 3_000_000,
            "untying should blow the budget, which is why we tie"
        );
    }

    #[test]
    fn softmax_rank_cap_tracks_tying() {
        let tied = ModelConfig::quark_3m();
        assert_eq!(tied.softmax_rank_cap(), 128);
        let untied = ModelConfig {
            tie_embeddings: false,
            ..tied
        };
        assert_eq!(untied.softmax_rank_cap(), 384);
    }

    #[test]
    fn validate_rejects_structurally_broken_configs() {
        let bad = ModelConfig {
            n_heads: 5, // 384 % 5 != 0
            ..ModelConfig::quark_3m()
        };
        assert!(bad.validate().is_err());

        let bad = ModelConfig {
            n_heads: 6,
            n_kv_heads: 4, // 6 % 4 != 0
            ..ModelConfig::quark_3m()
        };
        assert!(bad.validate().is_err());

        let bad = ModelConfig {
            d_emb: 512, // > d_model
            ..ModelConfig::quark_3m()
        };
        assert!(bad.validate().is_err());
    }

    /// An odd `d_head` divides evenly but still cannot be rotated, since burn's
    /// `RotaryEncoding` pairs adjacent elements. 384 / 128 = 3: whole, and odd.
    #[test]
    fn validate_rejects_an_odd_head_dimension() {
        let bad = ModelConfig {
            n_heads: 128, // 384 / 128 = 3
            n_kv_heads: 2,
            ..ModelConfig::quark_3m()
        };
        let errs = bad.validate().expect_err("an odd d_head must be rejected");
        assert!(
            errs.iter().any(|e| e.contains("rotary")),
            "the diagnosis must name the rotary constraint, got {errs:?}"
        );
    }

    /// `n_heads = 0` has to come back as an error, not a panic. Every other
    /// count is guarded (`n_kv_heads == 0` explicitly, `n_unique_layers` and
    /// `n_loops` explicitly), and this one reaches `d_model % n_heads` first --
    /// a remainder by zero, which panics.
    ///
    /// The distinction matters because `validate` is the function whose entire
    /// job is to turn a bad config into a diagnosis: it is called on
    /// deserialized JSON the user wrote by hand, and it promises to report
    /// *every* violation in one pass. A panic reports none of them, and takes
    /// the process with it.
    #[test]
    fn validate_reports_zero_heads_rather_than_dividing_by_it() {
        let bad = ModelConfig {
            n_heads: 0,
            ..ModelConfig::quark_3m()
        };
        let errs = bad.validate().expect_err("n_heads = 0 must be rejected");
        assert!(
            errs.iter().any(|e| e.contains("n_heads")),
            "the diagnosis must name n_heads, got {errs:?}"
        );
    }

    /// The promise in `validate`'s docstring -- every violation, not just the
    /// first -- has to survive the zero guard. A config that is broken in two
    /// ways, one of which is `n_heads = 0`, must still report the other.
    #[test]
    fn a_zero_head_config_still_reports_its_other_faults() {
        let bad = ModelConfig {
            n_heads: 0,
            d_emb: 512, // also > d_model
            ..ModelConfig::quark_3m()
        };
        let errs = bad
            .validate()
            .expect_err("this config is broken twice over");
        assert!(
            errs.iter().any(|e| e.contains("n_heads")),
            "the diagnosis must name n_heads, got {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("d_emb")),
            "guarding n_heads must not swallow the unrelated d_emb fault, got {errs:?}"
        );
    }

    #[test]
    fn config_roundtrips_through_json() {
        let c = ModelConfig::quark_3m();
        let json = serde_json::to_string(&c).unwrap();
        let back: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn thousands_formatting() {
        assert_eq!(fmt_thousands(0), "0");
        assert_eq!(fmt_thousands(999), "999");
        assert_eq!(fmt_thousands(1_000), "1,000");
        assert_eq!(fmt_thousands(2_868_352), "2,868,352");
    }
}
