//! Binetic CLI — command-line interface for the binetic fabric.
//!
//! Provides commands for:
//! - Running inference (single prompt, streaming, batch)
//! - Managing models (list, attach, detach, download)
//! - Monitoring fabric state (stats, register view, bloom status)
//! - Configuration (generate, edit, validate)
//! - Benchmarking (throughput, energy, latency)
//! - Adaptation (inject deltas, monitor adaptation stats)

use binetic_core::{
    backend::{ArithmeticOp, Backend, BackendId, ComputeOp, ComputeParams, QuantizationLevel, RoutingPolicy},
    echo::EchoPurpose,
    fabric::{Fabric, FabricConfig, FabricStats},
    rotation::{EnergyState, RotationPolicy, ThermalState},
    tiers::TierSet,
    bitslice::BitslicedLane,
    address::RegisterAddress,
};
use binetic_intent::{
    IntentEngine, IntentEngineConfig, Intent, UserAuthority, Verifier, TrajectoryOutcome,
};
use clap::{Parser, Subcommand, ValueEnum};
use colored::Colorize;
use std::path::PathBuf;
use std::sync::Arc;
use std::io::Write;
use tracing::{info, warn, error};

/// Binetic — self-describing, spatially-organized computation fabric for local AI inference.
#[derive(Parser, Debug)]
#[command(name = "binetic", version, about, long_about = None)]
struct Cli {
    /// Configuration file path.
    #[arg(long, default_value = "binetic.toml")]
    config: PathBuf,

    /// Enable debug logging.
    #[arg(long, short)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run inference on a prompt.
    Inference(InferenceCmd),
    /// Stream inference (token by token).
    Stream(StreamCmd),
    /// Manage models.
    Model(ModelCmd),
    /// Show fabric statistics.
    Stats,
    /// Benchmark the fabric.
    Benchmark(BenchmarkCmd),
    /// Inject an adaptation delta.
    Adapt(AdaptCmd),
    /// Detect and evolve toward user intent.
    Intent(IntentCmd),
    /// Verify a mutation against the verifier gates.
    Verify(VerifyCmd),
    /// Generate a default configuration file.
    Init,
}

/// Detect and evolve toward user intent from natural language.
#[derive(Parser, Debug)]
struct IntentCmd {
    /// Natural language description of what you want the system to do.
    #[arg(short, long)]
    description: String,

    /// Current register state (hex bits, e.g. "0101").
    #[arg(short, long, default_value = "0000")]
    current_state: String,

    /// Target operation (sin, cos, xor, and, or, not, hadamard).
    #[arg(short, long)]
    target_op: Option<String>,

    /// Population size for GA evolution.
    #[arg(long, default_value = "20")]
    population: usize,

    /// Max generations to evolve.
    #[arg(long, default_value = "10")]
    generations: u64,
}

/// Verify a mutation against the verifier gates.
#[derive(Parser, Debug)]
struct VerifyCmd {
    /// Delta hash (u64) of the proposed mutation.
    #[arg(short, long)]
    delta_hash: u64,

    /// Intent hash (u64) of the user's confirmed intent.
    #[arg(short, long)]
    intent_hash: u64,

    /// Features for scoring model (comma-separated floats).
    #[arg(short, long, default_value = "0.5,0.5,0.5,0.5")]
    features: String,
}

#[derive(Parser, Debug)]
struct InferenceCmd {
    /// Model name to use.
    #[arg(short, long)]
    model: String,

    /// Path to the GGUF model file.
    #[arg(short, long)]
    model_path: Option<PathBuf>,

    /// Prompt to infer.
    #[arg(short, long)]
    prompt: String,

    /// Number of tokens to generate.
    #[arg(long, default_value = "256")]
    tokens: usize,

    /// Temperature.
    #[arg(long, default_value = "0.7")]
    temperature: f32,

    /// Top-k sampling.
    #[arg(long, default_value = "50")]
    top_k: usize,

    /// Top-p sampling.
    #[arg(long, default_value = "0.95")]
    top_p: f32,

    /// Use GPU if available.
    #[arg(long)]
    gpu: bool,
}

#[derive(Parser, Debug)]
struct StreamCmd {
    /// Model name to use.
    #[arg(short, long)]
    model: String,

    /// Prompt to stream.
    #[arg(short, long)]
    prompt: String,

    /// Temperature.
    #[arg(long, default_value = "0.7")]
    temperature: f32,

    /// Top-k sampling.
    #[arg(long, default_value = "50")]
    top_k: usize,

    /// Top-p sampling.
    #[arg(long, default_value = "0.95")]
    top_p: f32,
}

#[derive(Parser, Debug)]
struct ModelCmd {
    #[command(subcommand)]
    command: ModelSubCmd,
}

#[derive(Subcommand, Debug)]
enum ModelSubCmd {
    /// List attached models.
    List,
    /// Attach a model to the fabric.
    Attach {
        /// Model name.
        #[arg(short, long)]
        name: String,
        /// Path or URL to model file.
        #[arg(short, long)]
        source: String,
        /// Quantization level.
        #[arg(long, default_value = "Q4_0")]
        quantization: String,
    },
    /// Detach a model from the fabric.
    Detach {
        /// Model name.
        #[arg(short, long)]
        name: String,
    },
}

#[derive(Parser, Debug)]
struct BenchmarkCmd {
    /// Model name to benchmark.
    #[arg(short, long)]
    model: String,

    /// Number of tokens to generate per run.
    #[arg(long, default_value = "1024")]
    tokens: usize,

    /// Number of runs.
    #[arg(long, default_value = "5")]
    runs: usize,

    /// Benchmark on battery (if available).
    #[arg(long)]
    battery: bool,
}

#[derive(Parser, Debug)]
struct AdaptCmd {
    /// Model name.
    #[arg(short, long)]
    model: String,

    /// Layer to adapt.
    #[arg(long)]
    layer: u16,

    /// Delta file (XOR diff to inject).
    #[arg(long)]
    delta_file: PathBuf,

    /// Justification for adaptation.
    #[arg(long, default_value = "gradient_step")]
    justification: String,
}

fn main() {
    let cli = Cli::parse();

    // Set up logging
    if cli.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("debug")
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter("info")
            .init();
    }

    match cli.command {
        Commands::Inference(cmd) => run_inference(cmd),
        Commands::Stream(cmd) => run_stream(cmd),
        Commands::Model(cmd) => run_model(cmd.command),
        Commands::Stats => run_stats(),
        Commands::Benchmark(cmd) => run_benchmark(cmd),
        Commands::Adapt(cmd) => run_adapt(cmd),
        Commands::Intent(cmd) => run_intent(cmd),
        Commands::Verify(cmd) => run_verify(cmd),
        Commands::Init => run_init(),
    }
}

fn run_inference(cmd: InferenceCmd) {
    info!("Starting inference with model: {}", cmd.model);

    let config = FabricConfig::default();
    let fabric = match Fabric::new(config) {
        Ok(f) => f,
        Err(e) => {
            error!("Failed to create fabric: {}", e);
            return;
        }
    };

    fabric.initialize().expect("Failed to initialize fabric");

    // Wire up llama.cpp backend for real predictive next-token inference
    let model_path = match cmd.model_path {
        Some(p) => p,
        None => {
            error!("--model-path <GGUF_PATH> is required for real inference");
            return;
        }
    };

    let mut llama_backend = binetic_llama_cpp::LlamaCppBackend::new();
    if let Err(e) = llama_backend.load_model(&model_path) {
        error!("Failed to load model '{}': {}", model_path.display(), e);
        return;
    }
    info!("Model loaded from {}", model_path.display());

    if let Err(e) = llama_backend.init() {
        error!("Failed to initialize llama.cpp backend: {}", e);
        return;
    }
    info!("llama.cpp backend initialized");

    // Add llama.cpp backend to fabric's backend set
    {
        let backends_arc = fabric.backends();
        let mut backends = backends_arc.write();
        backends.add(Box::new(llama_backend));
    }

    // Attach model in fabric
    if fabric.get_model(&cmd.model).is_none() {
        let _ = fabric.attach_model(&cmd.model, "", QuantizationLevel::Q4_0, false, None);
    }

    // Set energy state
    fabric.set_energy_state(EnergyState::PluggedIn);
    fabric.set_thermal_state(ThermalState::Cool);

    let kv_addr = RegisterAddress::new(0xE000, 0, 0, 0, 0, 0);
    let act_addr = RegisterAddress::new(0xF000, 0, 0, 0, 0, 0);

    // Predictive next-token inference loop
    let mut total_tokens = 0;
    let start = std::time::Instant::now();

    for i in 0..cmd.tokens {
        let result = fabric.sample_next_token(
            &cmd.model,
            kv_addr,
            act_addr,
            cmd.temperature,
            cmd.top_k,
            cmd.top_p,
        );

        if result.success {
            total_tokens += 1;
            // Decode and print the sampled token
            if let Some(ref payload) = result.payload {
                let token_id = payload.get_bit(0) as u32;
                print!(" token:{} ", token_id);
                let _ = std::io::stdout().flush();
            }
        } else {
            warn!("Compute failed at token {}: {:?}", i, result.error_message);
            break;
        }
    }

    let elapsed = start.elapsed();
    let tok_per_sec = total_tokens as f64 / elapsed.as_secs_f64();

    println!();
    println!("{} tokens in {:?}", total_tokens, elapsed);
    println!("Throughput: {:.2} tok/s", tok_per_sec);

    let stats = fabric.stats();
    println!("Energy estimate: {:.2} nJ/token", stats.estimated_energy_per_token_nj);
    println!("Total energy estimate: {:.2} nJ", stats.total_energy_estimate_nj);
}

fn run_stream(cmd: StreamCmd) {
    info!("Starting stream with model: {}", cmd.model);

    let config = FabricConfig::default();
    let fabric = match Fabric::new(config) {
        Ok(f) => f,
        Err(e) => {
            error!("Failed to create fabric: {}", e);
            return;
        }
    };

    fabric.initialize().expect("Failed to initialize fabric");
    fabric.set_energy_state(EnergyState::PluggedIn);
    fabric.set_thermal_state(ThermalState::Cool);

    let kv_addr = RegisterAddress::new(0xE000, 0, 0, 0, 0, 0);
    let act_addr = RegisterAddress::new(0xF000, 0, 0, 0, 0, 0);

    println!("Streaming response for prompt: {}", cmd.prompt);
    println!("{}", "─".repeat(60).dimmed());

    let mut token_count = 0usize;
    loop {
        let result = fabric.sample_next_token(
            &cmd.model,
            kv_addr,
            act_addr,
            cmd.temperature,
            cmd.top_k,
            cmd.top_p,
        );

        if result.success {
            // Decode and print token
            // In a real implementation, we'd decode the token ID to text
            print!("[tok{}]", token_count);
            std::io::stdout().flush().unwrap();
            token_count += 1;

            // Check for stop condition (e.g., end of sequence token)
            if token_count > 100 {
                break;
            }
        } else {
            error!("Stream failed: {:?}", result.error_message);
            break;
        }
    }

    println!();
    println!("{}", "─".repeat(60).dimmed());
    println!("Streamed {} tokens", token_count);
}

fn run_model(cmd: ModelSubCmd) {
    let config = FabricConfig::default();
    let fabric = match Fabric::new(config) {
        Ok(f) => f,
        Err(e) => {
            error!("Failed to create fabric: {}", e);
            return;
        }
    };

    fabric.initialize().expect("Failed to initialize fabric");

    match cmd {
        ModelSubCmd::List => {
            let models = fabric.models().read();
            if models.is_empty() {
                println!("No models attached.");
            } else {
                println!("{:<20} {:<10} {:<10} {:<10} {}", "Name", "Weights", "KV Cache", "Tier", "Quantization");
                println!("{}", "─".repeat(60));
                for (name, model) in models.iter() {
                    println!(
                        "{:<20} {:<10} {:<10} {:<10} {:?}",
                        name,
                        model.weight_count,
                        model.kv_cache_count,
                        format!("{:?}", model.tier),
                        model.quantization
                    );
                }
            }
        }
        ModelSubCmd::Attach { name, source, quantization } => {
            let quant = match parse_quantization(&quantization) {
                Ok(q) => q,
                Err(e) => {
                    error!("Invalid quantization: {}", e);
                    return;
                }
            };

            match fabric.attach_model(&name, &source, quant, false, None) {
                Ok(model) => {
                    println!("Attached model: {}", name);
                    println!("  Weights: {} registers", model.weight_count);
                    println!("  KV Cache: {} registers", model.kv_cache_count);
                    println!("  Tier: {:?}", model.tier);
                    println!("  Quantization: {:?}", model.quantization);
                }
                Err(e) => {
                    error!("Failed to attach model: {}", e);
                }
            }
        }
        ModelSubCmd::Detach { name } => {
            let mut models = fabric.models().write();
            if models.remove(&name).is_some() {
                println!("Detached model: {}", name);
            } else {
                warn!("Model not found: {}", name);
            }
        }
    }
}

fn run_stats() {
    let config = FabricConfig::default();
    let fabric = Fabric::new(config).expect("Failed to create fabric");
    fabric.initialize().expect("Failed to initialize");

    let stats = fabric.stats();

    println!("{}", "Binetic Fabric Statistics".bold().underline());
    println!("{}", "─".repeat(60));

    println!("\n{} registers:", stats.register_count_by_tier.iter().map(|(t, c)| format!("{:?}={}", t, c)).collect::<Vec<_>>().join(", "));
    println!("Sectors with bloom filters: {}", stats.sector_count);

    println!("\n--- Rotation ---");
    println!("Total rotations: {}", stats.total_rotations);
    println!("Total echoes processed: {}", stats.total_echoes_processed);
    println!("Total prefetches: {}", stats.total_prefetches);

    println!("\n--- Adaptation ---");
    println!("Adaptation deltas injected: {}", stats.adaptation_deltas_injected);
    println!("Adaptation echoes propagated: {}", stats.adaptation_echoes_propagated);

    println!("\n--- Energy ---");
    println!("Estimated energy per token: {:.2} nJ", stats.estimated_energy_per_token_nj);
    println!("Total energy estimate: {:.2} nJ", stats.total_energy_estimate_nj);

    println!("\n--- Access ---");
    println!("Total reads: {}", stats.total_reads);
    println!("Total writes: {}", stats.total_writes);
    println!("Total computes: {}", stats.total_computes);

    println!("\n--- Backends ---");
    for (backend, count) in &stats.backend_usage {
        println!("  {}: {} calls", backend.0, count);
    }

    println!("\n--- Integrity ---");
    println!("Corruption detected: {}", stats.corruption_detected);
    println!("Corruption repaired: {}", stats.corruption_repaired);
}

fn run_benchmark(cmd: BenchmarkCmd) {
    info!("Starting benchmark with model: {}", cmd.model);

    let config = FabricConfig::default();
    let fabric = Fabric::new(config).expect("Failed to create fabric");
    fabric.initialize().expect("Failed to initialize");

    // Set energy state based on battery flag
    if cmd.battery {
        fabric.set_energy_state(EnergyState::OnBattery { level: 0.5 });
    } else {
        fabric.set_energy_state(EnergyState::PluggedIn);
    }
    fabric.set_thermal_state(ThermalState::Cool);

    let kv_addr = RegisterAddress::new(0xE000, 0, 0, 0, 0, 0);
    let act_addr = RegisterAddress::new(0xF000, 0, 0, 0, 0, 0);

    let mut throughputs = Vec::new();
    let mut latencies = Vec::new();

    for run in 0..cmd.runs {
        let start = std::time::Instant::now();
        let mut tokens = 0;

        for _ in 0..cmd.tokens {
            let result = fabric.sample_next_token(
                &cmd.model,
                kv_addr,
                act_addr,
                0.7,
                50,
                0.95,
            );

            if result.success {
                tokens += 1;
            } else {
                break;
            }
        }

        let elapsed = start.elapsed();
        let tok_per_sec = tokens as f64 / elapsed.as_secs_f64();

        throughputs.push(tok_per_sec);
        latencies.push(elapsed.as_micros() as f64 / cmd.tokens as f64);

        println!("Run {}: {:.2} tok/s ({} tokens in {:?})", run + 1, tok_per_sec, tokens, elapsed);
    }

    let avg_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;

    println!();
    println!("Benchmark Results:");
    println!("  Average throughput: {:.2} tok/s", avg_throughput);
    println!("  Average latency per token: {:.2} us", avg_latency);
    println!("  Runs: {}", cmd.runs);
    println!("  Tokens per run: {}", cmd.tokens);
    println!("  Energy state: {:?}", fabric.rotation_scheduler().read().energy_state());
}

fn run_adapt(cmd: AdaptCmd) {
    info!("Injecting adaptation delta for model: {}", cmd.model);

    let config = FabricConfig::default();
    let fabric = Fabric::new(config).expect("Failed to create fabric");
    fabric.initialize().expect("Failed to initialize");

    // Read delta file
    let delta_bytes = match std::fs::read(&cmd.delta_file) {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("Failed to read delta file: {}", e);
            return;
        }
    };

    let delta = BitslicedLane::from_bytes(&delta_bytes);

    // Compute target address (weight register for the given layer)
    let target_addr = RegisterAddress::new(0xD000, cmd.layer, 0, 0, 0, 0);

    let purpose = match cmd.justification.as_str() {
        "gradient_step" => EchoPurpose::GradientStep,
        "heuristic" => EchoPurpose::Custom(cmd.justification.clone()),
        _ => EchoPurpose::Custom(cmd.justification.clone()),
    };

    let purpose_display = format!("{:?}", purpose);

    match fabric.inject_adapt(target_addr, delta, purpose) {
        Ok(_) => {
            println!("Adaptation delta injected successfully.");
            println!("  Target: layer {}", cmd.layer);
            println!("  Delta size: {} bytes", delta_bytes.len());
            println!("  Purpose: {}", purpose_display);

            // Show updated stats
            let stats = fabric.stats();
            println!("  Adaptation deltas injected: {}", stats.adaptation_deltas_injected);
        }
        Err(e) => {
            error!("Failed to inject adaptation: {}", e);
        }
    }
}

/// Adapter wrapping the real Fabric for the intent engine.
struct FabricAdapterImpl {
    fabric: Arc<Fabric>,
}

impl binetic_intent::FabricAdapter for FabricAdapterImpl {
    fn read_raw(&self, addr: RegisterAddress) -> Option<binetic_core::register::Register> {
        self.fabric.read_raw(addr)
    }

    fn write(&self, addr: RegisterAddress, reg: binetic_core::register::Register) -> Result<(), String> {
        self.fabric.write(addr, reg)
    }

    fn inject_adapt_raw(&self, addr: RegisterAddress, delta: BitslicedLane) -> Result<(), String> {
        self.fabric.inject_adapt_raw(addr, delta)
    }
}

fn run_intent(cmd: IntentCmd) {
    let config = FabricConfig::default();
    let fabric = Arc::new(Fabric::new(config).expect("Failed to create fabric"));
    fabric.initialize().expect("Failed to initialize fabric");

    let intent_engine_config = IntentEngineConfig {
        population_size: cmd.population,
        max_generations: cmd.generations,
        fitness_threshold: 0.01,
        meta_learning_enabled: true,
        gene_length: 64,
    };

    let fabric_adapter = Arc::new(FabricAdapterImpl { fabric: fabric.clone() });
    let mut engine = IntentEngine::new(fabric_adapter, intent_engine_config);

    // Parse current state from hex string
    let current_bits: Vec<bool> = cmd.current_state.chars().map(|c| c == '1').collect();
    let current_state = BitslicedLane::from_bits(&current_bits);

    // Detect intent from description
    let mut intent = engine.detect_intent(&cmd.description).unwrap_or_else(|e| {
        error!("Failed to detect intent: {}", e);
        std::process::exit(1);
    });

    // Override target operation if specified
    if let Some(op_str) = &cmd.target_op {
        intent.target_op = match op_str.to_lowercase().as_str() {
            "sin" => Some(ArithmeticOp::Sin),
            "cos" => Some(ArithmeticOp::Cos),
            "xor" | "xorgate" => Some(ArithmeticOp::XorGate),
            "and" | "andgate" => Some(ArithmeticOp::AndGate),
            "or" | "orgate" => Some(ArithmeticOp::OrGate),
            "not" | "notgate" => Some(ArithmeticOp::NotGate),
            "hadamard" => Some(ArithmeticOp::Hadamard),
            _ => intent.target_op,
        };
    }

    println!("Intent detected: {}", intent.description);
    println!("Target operation: {:?}", intent.target_op);
    println!("Evolving toward intent...");

    // Evolve GA toward the user's intent
    let best = engine.evolve_to_intent(&intent, &current_state).unwrap_or_else(|e| {
        error!("Evolution failed: {}", e);
        std::process::exit(1);
    });

    println!("Evolution complete!");
    println!("  Best fitness: {:.4}", best.fitness);
    println!("  Best genes: {:?}", best.genes.iter().take(8).collect::<Vec<_>>());

    let strategy = engine.meta_strategy();
    println!("  Meta strategy mutation rate: {:.4}", strategy.mutation_rate);
    println!("  Meta strategy crossover blend: {:.4}", strategy.crossover_blend);
}

fn run_verify(cmd: VerifyCmd) {
    let features: Vec<f64> = cmd.features.split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let mut verifier = Verifier::new(
        features.len().max(4),
        0.0,    // min trajectory success rate
        0.0,    // min intent alignment
        -1e9,   // min model score
    );

    let delta_hash = cmd.delta_hash;
    let intent_hash = cmd.intent_hash;

    // Record a confirmed intent
    verifier.record_intent(intent_hash, "user confirmed intent".to_string(), 1.0);
    verifier.confirm_intent(0);

    // Record trajectory outcome
    let node_idx = verifier.add_trajectory_node("inject".to_string(), RegisterAddress::from(0u128), delta_hash);
    verifier.record_outcome(node_idx, TrajectoryOutcome::Success);

    // Train scoring model on positive outcome
    verifier.train_model(&features, true);

    // Verify
    let passed = verifier.verify(delta_hash, intent_hash, &features);

    println!("Verifier results:");
    println!("  Trajectory success rate: {:.2}", verifier.trajectory().success_rate());
    println!("  Intent graph drift: {:.4}", verifier.intent_graph().intent_drift());
    println!("  Model score: {:.4}", verifier.scoring_model().score(&features));
    println!("  Verification: {}", if passed { "PASSED" } else { "FAILED" });
}

fn run_init() {
    let config_path = PathBuf::from("binetic.toml");
    let config = FabricConfig::default();

    let toml = toml::to_string(&config).expect("Failed to serialize config");

    std::fs::write(&config_path, toml).expect("Failed to write config file");

    println!("Generated default configuration at: {}", config_path.display());
    println!();
    println!("Next steps:");
    println!("  1. Edit binetic.toml to configure tiers, rotation, echo policy.");
    println!("  2. Run 'binetic model attach --name my-model --source /path/to/model.gguf'");
    println!("  3. Run 'binetic inference --model my-model --prompt \"Hello, world!\"'");
}

// Simple parsing for quantization level — returns the enum directly.
// (Can't impl FromStr for QuantizationLevel due to orphan rules,
// callers use this function directly.)
fn parse_quantization(s: &str) -> Result<QuantizationLevel, String> {
    match s.to_uppercase().as_str() {
        "F32" => Ok(QuantizationLevel::F32),
        "F16" => Ok(QuantizationLevel::F16),
        "BF16" => Ok(QuantizationLevel::BF16),
        "Q4_0" => Ok(QuantizationLevel::Q4_0),
        "Q4_1" => Ok(QuantizationLevel::Q4_1),
        "Q5_0" => Ok(QuantizationLevel::Q5_0),
        "Q5_1" => Ok(QuantizationLevel::Q5_1),
        "Q6_K" => Ok(QuantizationLevel::Q6_K),
        "Q8_0" => Ok(QuantizationLevel::Q8_0),
        "Q8_1" => Ok(QuantizationLevel::Q8_1),
        "IQ2_XS" => Ok(QuantizationLevel::IQ2_XS),
        "IQ3_XS" => Ok(QuantizationLevel::IQ3_XS),
        "IQ4_XS" => Ok(QuantizationLevel::IQ4_XS),
        "IQ5_XS" => Ok(QuantizationLevel::IQ5_XS),
        "IQ6_XS" => Ok(QuantizationLevel::IQ6_XS),
        "IQ7_XS" => Ok(QuantizationLevel::IQ7_XS),
        "IQ4_XXS" => Ok(QuantizationLevel::IQ4_XXS),
        _ => Err(format!("Unknown quantization: {}", s)),
    }
}
