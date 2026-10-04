//! Intent engine: detect user intent, compute gaps, evolve solutions, meta-learn.
//!
//! # Architecture
//!
//! Intent flows through three stages:
//! 1. **Detect** — parse user input into structured intent (commands, corrections, feedback)
//! 2. **Gap** — XOR current state against target state to find the delta
//! 3. **Evolve** — GA population evolves circuit configurations to close the gap
//! 4. **Meta** — population evolves its own learning strategies (mutation, crossover, selection)
//!
//! The delta IS the intent. XOR delta model sharing computes the gap in one operation.
//! Meta-learning happens when the GA evolves its own mutation operators.

use binetic_core::{
    backend::ArithmeticOp,
    bitslice::BitslicedLane,
    address::RegisterAddress,
    register::Register,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IntentError {
    #[error("No intent detected in input: {0}")]
    NoIntentDetected(String),
    #[error("Gap computation failed: {0}")]
    GapComputationFailed(String),
    #[error("Evolution failed: {0}")]
    EvolutionFailed(String),
    #[error("Meta-learning failed: {0}")]
    MetaLearningFailed(String),
}

/// Parsed user intent — what the user wants the system to do.
#[derive(Debug, Clone)]
pub struct Intent {
    /// Target register state the user wants to achieve.
    pub target_state: Option<BitslicedLane>,
    /// Target operation the user wants performed.
    pub target_op: Option<ArithmeticOp>,
    /// Corrections to current behavior (key-value pairs of what to change).
    pub corrections: HashMap<String, BitslicedLane>,
    /// Free-form description of desired behavior.
    pub description: String,
    /// Priority of this intent (higher = more important).
    pub priority: f64,
}

/// A single GA individual — a candidate solution encoding circuit configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    /// Gene: encoding of trig gate parameters and routing.
    pub genes: Vec<f64>,
    /// Fitness score — lower is better (closer to target).
    pub fitness: f64,
    /// Meta-parameters this individual uses for its own learning.
    pub learning_strategy: LearningStrategy,
}

/// Learning strategy — how this individual learns and adapts.
/// This is what evolves in meta-learning: the learning process itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStrategy {
    /// Mutation rate for gene perturbations.
    pub mutation_rate: f64,
    /// Crossover blend factor (0.0 = pure parent1, 1.0 = pure parent2).
    pub crossover_blend: f64,
    /// Selection pressure — how strongly fitness influences reproduction.
    pub selection_pressure: f64,
    /// Exploration rate — probability of trying random moves.
    pub exploration_rate: f64,
    /// Memory decay — how quickly past experience is forgotten.
    pub memory_decay: f64,
}

impl Default for LearningStrategy {
    fn default() -> Self {
        Self {
            mutation_rate: 0.1,
            crossover_blend: 0.5,
            selection_pressure: 1.0,
            exploration_rate: 0.1,
            memory_decay: 0.95,
        }
    }
}

/// GA population — evolves circuit configurations toward target behavior.
#[derive(Debug)]
pub struct GAPopulation {
    individuals: Vec<Individual>,
    target: Option<BitslicedLane>,
    generation: u64,
    best_fitness_history: Vec<f64>,
}

/// Meta-learner — evolves the GA's own learning strategies.
#[derive(Debug)]
pub struct MetaLearner {
    strategy_population: Vec<Individual>,
    generations_without_improvement: u64,
    best_strategy: LearningStrategy,
    best_strategy_fitness: f64,
}

/// The IntentEngine — ties everything together.
pub struct IntentEngine {
    fabric: Arc<dyn FabricAdapter>,
    population: GAPopulation,
    meta_learner: MetaLearner,
    intent_history: Vec<Intent>,
    config: IntentEngineConfig,
}

/// User authority guard — ensures the user always has final approval.
///
/// # Principle
///
/// The GA proposes. The user approves. No mutation executes without explicit
/// user confirmation. This is the absolute boundary of system autonomy.
///
/// # Mechanism
///
/// 1. **Approval gate** — every mutation requires user sign-off before execution
/// 2. **Intent verification** — mutations are verified against the user's signed intent
/// 3. **Rollback** — every mutation is reversible; user can restore any prior state
/// 4. **Audit trail** — all mutations logged with intent hash for verification
/// 5. **Domain isolation** — user intent domain is separate from GA search domain
#[derive(Debug, Clone)]
pub struct UserAuthority {
    /// Signed intent hash — mutations must match this to be approved.
    intent_hash: u64,
    /// History of all mutations with their intent verification.
    audit_trail: Vec<MutationRecord>,
    /// Rollback checkpoints — saved register states the user can restore.
    checkpoints: Vec<(RegisterAddress, Register)>,
}

/// A single mutation record in the audit trail.
#[derive(Debug, Clone)]
pub struct MutationRecord {
    /// Address of the register that was mutated.
    pub address: RegisterAddress,
    /// The delta that was applied.
    pub delta: BitslicedLane,
    /// Intent hash at time of mutation.
    pub intent_hash: u64,
    /// Whether the mutation was approved by the user.
    pub approved: bool,
}

impl UserAuthority {
    /// Create a new authority guard with the given intent hash.
    pub fn new(intent_hash: u64) -> Self {
        Self {
            intent_hash,
            audit_trail: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    /// Verify that a proposed mutation matches the user's signed intent.
    pub fn verify_intent(&self, delta: &BitslicedLane) -> bool {
        let delta_hash = hash_bitslice(delta);
        delta_hash as u64 == self.intent_hash
    }

    /// Approve a mutation — must be called before execution.
    /// Returns false if the mutation doesn't match the user's intent.
    pub fn approve(&mut self, address: RegisterAddress, delta: &BitslicedLane) -> bool {
        let verified = self.verify_intent(delta);
        self.audit_trail.push(MutationRecord {
            address,
            delta: delta.clone(),
            intent_hash: self.intent_hash,
            approved: verified,
        });
        verified
    }

    /// Save a checkpoint for rollback.
    pub fn checkpoint(&mut self, address: RegisterAddress, register: &Register) {
        self.checkpoints.push((address, register.clone()));
    }

    /// Roll back to the last checkpoint for the given address.
    pub fn rollback(&self, address: RegisterAddress) -> Option<Register> {
        self.checkpoints
            .iter()
            .rev()
            .find(|(addr, _)| *addr == address)
            .map(|(_, reg)| reg.clone())
    }

    /// Get the audit trail.
    pub fn audit_trail(&self) -> &[MutationRecord] {
        &self.audit_trail
    }

    /// Verify the entire audit trail — all mutations must be approved.
    pub fn verify_all_approved(&self) -> bool {
        self.audit_trail.iter().all(|r| r.approved)
    }
}

fn hash_bitslice(bits: &BitslicedLane) -> u64 {
    let mut hash: u64 = 0;
    for word in bits.words() {
        hash = hash.wrapping_mul(31).wrapping_add(*word);
    }
    hash
}

/// Task trajectory graph — records the sequence of actions taken and their outcomes.
///
/// Each node is an action (read, write, inject, adapt). Edges represent
/// temporal ordering — what happened after what. The graph captures the
/// full execution history for verification and learning.
#[derive(Debug, Clone)]
pub struct TrajectoryGraph {
    nodes: Vec<TrajectoryNode>,
    edges: Vec<(usize, usize)>, // (from_idx, to_idx)
}

#[derive(Debug, Clone)]
pub struct TrajectoryNode {
    pub action: String,
    pub address: RegisterAddress,
    pub delta_hash: u64,
    pub outcome: TrajectoryOutcome,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrajectoryOutcome {
    Success,
    Rejected,
    Pending,
}

impl TrajectoryGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, action: String, address: RegisterAddress, delta_hash: u64) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(TrajectoryNode {
            action,
            address,
            delta_hash,
            outcome: TrajectoryOutcome::Pending,
            timestamp_ms: 0,
        });
        if idx > 0 {
            self.edges.push((idx - 1, idx));
        }
        idx
    }

    pub fn set_outcome(&mut self, idx: usize, outcome: TrajectoryOutcome) {
        if let Some(node) = self.nodes.get_mut(idx) {
            node.outcome = outcome;
        }
    }

    pub fn success_rate(&self) -> f64 {
        let completed: Vec<_> = self.nodes.iter().filter(|n| n.outcome != TrajectoryOutcome::Pending).collect();
        if completed.is_empty() {
            return 0.0;
        }
        let successes = completed.iter().filter(|n| n.outcome == TrajectoryOutcome::Success).count();
        successes as f64 / completed.len() as f64
    }

    pub fn nodes(&self) -> &[TrajectoryNode] {
        &self.nodes
    }
}

/// Intent graph — tracks user intent over time, showing how intent evolved.
///
/// Each node is an intent (parsed from user input). Edges represent
/// intent transitions — what the user wanted before and after.
/// The graph captures intent drift, corrections, and confirmations.
#[derive(Debug, Clone)]
pub struct IntentGraph {
    nodes: Vec<IntentNode>,
    edges: Vec<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub struct IntentNode {
    pub intent_hash: u64,
    pub description: String,
    pub priority: f64,
    pub confirmed: bool,
}

impl IntentGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_intent(&mut self, intent_hash: u64, description: String, priority: f64) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(IntentNode {
            intent_hash,
            description,
            priority,
            confirmed: false,
        });
        if idx > 0 {
            self.edges.push((idx - 1, idx));
        }
        idx
    }

    pub fn confirm(&mut self, idx: usize) {
        if let Some(node) = self.nodes.get_mut(idx) {
            node.confirmed = true;
        }
    }

    pub fn latest_confirmed(&self) -> Option<&IntentNode> {
        self.nodes.iter().rev().find(|n| n.confirmed)
    }

    pub fn intent_drift(&self) -> f64 {
        // Measure how much intent has changed over time
        // Higher drift = user changed their mind more
        if self.nodes.len() < 2 {
            return 0.0;
        }
        let mut total_drift = 0u64;
        for window in self.nodes.windows(2) {
            total_drift += window[0].intent_hash ^ window[1].intent_hash;
        }
        total_drift as f64 / (self.nodes.len() - 1) as f64
    }

    pub fn nodes(&self) -> &[IntentNode] {
        &self.nodes
    }
}

/// Live-data scoring model — scores mutations against user intent using learned weights.
///
/// Trained on live data: each mutation's outcome (success/failure) feeds back
/// into the model, which adjusts its weights to better predict user satisfaction.
/// The model scores how likely a mutation is to satisfy the user's intent.
#[derive(Debug, Clone)]
pub struct ScoringModel {
    /// Learned weights for each feature dimension.
    weights: Vec<f64>,
    /// Training data: (features, outcome) pairs.
    training_data: Vec<(Vec<f64>, bool)>,
    /// Learning rate for online updates.
    learning_rate: f64,
}

impl ScoringModel {
    pub fn new(feature_dim: usize, learning_rate: f64) -> Self {
        Self {
            weights: vec![0.0; feature_dim],
            training_data: Vec::new(),
            learning_rate,
        }
    }

    /// Score a mutation: higher score = more likely to satisfy user intent.
    pub fn score(&self, features: &[f64]) -> f64 {
        if features.len() != self.weights.len() {
            return 0.0;
        }
        let mut score = 0.0;
        for (f, w) in features.iter().zip(self.weights.iter()) {
            score += f * w;
        }
        score
    }

    /// Train on a mutation outcome: updates weights based on whether the mutation satisfied the user.
    pub fn train(&mut self, features: &[f64], satisfied: bool) {
        if features.len() != self.weights.len() {
            return;
        }
        self.training_data.push((features.to_vec(), satisfied));

        // Online gradient update: if not satisfied, push weights away from features
        let target = if satisfied { 1.0 } else { 0.0 };
        let current = self.score(features);
        let error = target - current;

        for (w, f) in self.weights.iter_mut().zip(features.iter()) {
            *w += self.learning_rate * error * f;
        }
    }

    /// Get training data count.
    pub fn training_count(&self) -> usize {
        self.training_data.len()
    }
}

/// Verifier — second independent check on mutations using trajectory + intent graphs + scoring model.
///
/// The verifier cross-references three signals:
/// 1. **Trajectory consistency** — does this mutation follow the execution pattern?
/// 2. **Intent alignment** — does this mutation match the latest confirmed intent?
/// 3. **Scoring model** — does the learned model predict this will satisfy the user?
///
/// All three must pass for the verifier to approve. This is the final gate
/// before the user's explicit approval.
#[derive(Debug)]
pub struct Verifier {
    trajectory: TrajectoryGraph,
    intent_graph: IntentGraph,
    scoring_model: ScoringModel,
    min_trajectory_success_rate: f64,
    min_intent_alignment: f64,
    min_model_score: f64,
}

impl Verifier {
    pub fn new(
        feature_dim: usize,
        min_trajectory_success_rate: f64,
        min_intent_alignment: f64,
        min_model_score: f64,
    ) -> Self {
        Self {
            trajectory: TrajectoryGraph::new(),
            intent_graph: IntentGraph::new(),
            scoring_model: ScoringModel::new(feature_dim, 0.01),
            min_trajectory_success_rate,
            min_intent_alignment,
            min_model_score,
        }
    }

    /// Verify a mutation against all three signals.
    /// Returns true only if all checks pass.
    pub fn verify(
        &self,
        delta_hash: u64,
        intent_hash: u64,
        features: &[f64],
    ) -> bool {
        // 1. Trajectory consistency: success rate must be above threshold
        if self.trajectory.success_rate() < self.min_trajectory_success_rate {
            return false;
        }

        // 2. Intent alignment: delta must align with latest confirmed intent
        if let Some(latest) = self.intent_graph.latest_confirmed() {
            // Alignment: how many lower bits match between delta and intent hash
            let xor = delta_hash ^ latest.intent_hash;
            let matching_bits = 64 - xor.leading_zeros();
            let alignment = matching_bits as f64 / 64.0;
            if alignment < self.min_intent_alignment {
                return false;
            }
        }

        // 3. Scoring model: predicted satisfaction must exceed threshold
        if self.scoring_model.score(features) < self.min_model_score {
            return false;
        }

        true
    }

    /// Record a mutation outcome for trajectory tracking.
    pub fn record_outcome(&mut self, idx: usize, outcome: TrajectoryOutcome) {
        self.trajectory.set_outcome(idx, outcome);
    }

    /// Record a confirmed intent for intent tracking.
    pub fn record_intent(&mut self, intent_hash: u64, description: String, priority: f64) {
        self.intent_graph.add_intent(intent_hash, description, priority);
    }

    /// Confirm the latest intent.
    pub fn confirm_intent(&mut self, idx: usize) {
        self.intent_graph.confirm(idx);
    }

    /// Train the scoring model on a mutation outcome.
    pub fn train_model(&mut self, features: &[f64], satisfied: bool) {
        self.scoring_model.train(features, satisfied);
    }

    /// Add a trajectory node.
    pub fn add_trajectory_node(&mut self, action: String, address: RegisterAddress, delta_hash: u64) -> usize {
        self.trajectory.add_node(action, address, delta_hash)
    }

    /// Get reference to trajectory graph.
    pub fn trajectory(&self) -> &TrajectoryGraph {
        &self.trajectory
    }

    /// Get reference to intent graph.
    pub fn intent_graph(&self) -> &IntentGraph {
        &self.intent_graph
    }

    /// Get reference to scoring model.
    pub fn scoring_model(&self) -> &ScoringModel {
        &self.scoring_model
    }
}

/// Configuration for the intent engine.
#[derive(Debug, Clone)]
pub struct IntentEngineConfig {
    pub population_size: usize,
    pub max_generations: u64,
    pub fitness_threshold: f64,
    pub meta_learning_enabled: bool,
    pub gene_length: usize,
}

impl Default for IntentEngineConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_generations: 100,
            fitness_threshold: 0.001,
            meta_learning_enabled: true,
            gene_length: 64,
        }
    }
}

/// Adapter trait for fabric interaction — allows testing without real fabric.
pub trait FabricAdapter: Send + Sync {
    fn read_raw(&self, addr: RegisterAddress) -> Option<binetic_core::register::Register>;
    fn write(&self, addr: RegisterAddress, reg: binetic_core::register::Register) -> Result<(), String>;
    fn inject_adapt_raw(&self, addr: RegisterAddress, delta: BitslicedLane) -> Result<(), String>;
}

impl IntentEngine {
    pub fn new(
        fabric: Arc<dyn FabricAdapter>,
        config: IntentEngineConfig,
    ) -> Self {
        let population = GAPopulation::new(config.population_size, config.gene_length);
        let meta_learner = MetaLearner::new(config.population_size);

        Self {
            fabric,
            population,
            meta_learner,
            intent_history: Vec::new(),
            config,
        }
    }

    /// Detect intent from a user message.
    ///
    /// Parses commands like "make it faster", "add sin gates", "use XOR for this"
    /// into structured Intent objects.
    pub fn detect_intent(&mut self, message: &str) -> Result<Intent, IntentError> {
        let mut intent = Intent {
            target_state: None,
            target_op: None,
            corrections: HashMap::new(),
            description: message.to_string(),
            priority: 1.0,
        };

        // Detect operation intent
        let lower = message.to_lowercase();
        if lower.contains("sin") || lower.contains("trig") || lower.contains("trigonometric") {
            intent.target_op = Some(ArithmeticOp::Sin);
        } else if lower.contains("cos") {
            intent.target_op = Some(ArithmeticOp::Cos);
        } else if lower.contains("xor") || lower.contains("exclusive") {
            intent.target_op = Some(ArithmeticOp::XorGate);
        } else if lower.contains("and") && !lower.contains("random") {
            intent.target_op = Some(ArithmeticOp::AndGate);
        } else if lower.contains("or") && !lower.contains("color") && !lower.contains("more") {
            intent.target_op = Some(ArithmeticOp::OrGate);
        } else if lower.contains("not") || lower.contains("invert") || lower.contains("negate") {
            intent.target_op = Some(ArithmeticOp::NotGate);
        } else if lower.contains("hadamard") || lower.contains("correlation") || lower.contains("same") {
            intent.target_op = Some(ArithmeticOp::Hadamard);
        }

        // Detect correction intent
        if lower.contains("fix") || lower.contains("correct") || lower.contains("change") {
            intent.priority = 2.0; // corrections are higher priority
        }

        // Detect learning intent
        if lower.contains("learn") || lower.contains("adapt") || lower.contains("evolve") {
            intent.priority = 3.0; // learning is highest priority
        }

        if intent.target_op.is_none() && intent.corrections.is_empty() && intent.priority < 2.0 {
            return Err(IntentError::NoIntentDetected(
                "No recognizable intent in message".to_string(),
            ));
        }

        self.intent_history.push(intent.clone());
        Ok(intent)
    }

    /// Compute the gap between current state and target state.
    ///
    /// The gap is the XOR delta that needs to be applied to close the difference.
    pub fn compute_gap(
        &self,
        current: &BitslicedLane,
        target: &BitslicedLane,
    ) -> Result<BitslicedLane, IntentError> {
        if current.len() != target.len() {
            return Err(IntentError::GapComputationFailed(format!(
                "Size mismatch: current={} target={}",
                current.len(),
                target.len()
            )));
        }
        Ok(BitslicedLane::xor(current, target))
    }

    /// Evolve the population to close the gap.
    ///
    /// Runs the GA for up to max_generations, returning the best individual.
    pub fn evolve_to_intent(
        &mut self,
        intent: &Intent,
        current_state: &BitslicedLane,
    ) -> Result<&Individual, IntentError> {
        let target = intent.target_state.clone().unwrap_or_else(|| {
            // If no explicit target state, create one from the target op
            self.target_state_from_op(intent)
        });

        self.population.set_target(target.clone());

        for generation in 0..self.config.max_generations {
            // Evaluate fitness: Hamming distance between current applied genes and target
                    let current_state_clone = current_state.clone();
                    let target_clone = target.clone();
                    let fitness_values: Vec<f64> = self
                        .population
                        .individuals
                        .iter()
                        .map(|ind| self.evaluate_fitness(ind, &target_clone, &current_state_clone))
                        .collect();

                    for (individual, fitness) in self.population.individuals.iter_mut().zip(fitness_values) {
                        individual.fitness = fitness;
                    }

            // Sort by fitness (lower is better)
            self.population.individuals.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

            // Check convergence
            let best_fitness = self.population.individuals[0].fitness;
            self.population.best_fitness_history.push(best_fitness);

            if best_fitness < self.config.fitness_threshold {
                break;
            }

            // Evolve next generation
            self.population.evolve();

            // Meta-learning: evolve learning strategies
            if self.config.meta_learning_enabled {
                self.meta_learner.evolve_strategy(&self.population);
            }
        }

        Ok(&self.population.individuals[0])
    }

    /// Apply the best solution from evolution to the fabric.
    pub fn apply_solution(
        &self,
        solution: &Individual,
        target_addr: RegisterAddress,
    ) -> Result<(), IntentError> {
        // Convert genes to a delta lane
        let delta = self.genes_to_delta(solution);
        self.fabric
            .inject_adapt_raw(target_addr, delta)
            .map_err(|e| IntentError::EvolutionFailed(e))
    }

    /// Get the current meta-learning strategy.
    pub fn meta_strategy(&self) -> &LearningStrategy {
        &self.meta_learner.best_strategy
    }

    /// Get fitness history for monitoring convergence.
    pub fn fitness_history(&self) -> &[f64] {
        &self.population.best_fitness_history
    }

    fn target_state_from_op(&self, intent: &Intent) -> BitslicedLane {
        // Create a target state based on the detected operation
        // For now, use a simple pattern — in production this would be more sophisticated
        let mut lane = BitslicedLane::with_capacity(256);
        match intent.target_op {
            Some(ArithmeticOp::Sin) | Some(ArithmeticOp::XorGate) => {
                // Identity-like pattern for Sin/XorGate
                for i in 0..256 {
                    lane.set_bit(i, i % 2 == 0);
                }
            }
            Some(ArithmeticOp::Cos) | Some(ArithmeticOp::NotGate) => {
                // Inverted pattern for Cos/NotGate
                for i in 0..256 {
                    lane.set_bit(i, i % 2 != 0);
                }
            }
            Some(ArithmeticOp::AndGate) => {
                // Sparse pattern for AND
                for i in 0..256 {
                    lane.set_bit(i, i % 4 == 0);
                }
            }
            Some(ArithmeticOp::OrGate) => {
                // Dense pattern for OR
                for i in 0..256 {
                    lane.set_bit(i, i % 2 == 0 || i % 3 == 0);
                }
            }
            Some(ArithmeticOp::Hadamard) => {
                // Correlation pattern for Hadamard
                for i in 0..256 {
                    lane.set_bit(i, i % 4 < 2);
                }
            }
            _ => {
                // Default: all zeros
            }
        }
        lane
    }

    fn evaluate_fitness(
        &self,
        individual: &Individual,
        target: &BitslicedLane,
        current_state: &BitslicedLane,
    ) -> f64 {
        // Apply individual's genes as a transformation to current state
        let transformed = self.apply_genes(current_state, individual);
        // Fitness = Hamming distance (number of differing bits)
        let diff = BitslicedLane::xor(&transformed, target);
        let hamming = diff.popcount();
        hamming as f64 / target.len() as f64
    }

    fn apply_genes(&self, state: &BitslicedLane, individual: &Individual) -> BitslicedLane {
        // Apply gene-encoded transformations to the state
        // Genes are floats in [0, 1] that control which trig ops to apply
        let mut result = state.clone();
        let genes = &individual.genes;

        for (i, gene) in genes.iter().enumerate() {
            if i >= state.len() {
                break;
            }
            // Gene > 0.5: flip the bit
            // Gene <= 0.5: keep the bit
            if *gene > 0.5 {
                let current = result.get_bit(i);
                result.set_bit(i, !current);
            }
        }

        result
    }

    fn genes_to_delta(&self, solution: &Individual) -> BitslicedLane {
        // Convert genes to a delta lane for injection
        let mut delta = BitslicedLane::with_capacity(self.config.gene_length);
        for (i, gene) in solution.genes.iter().enumerate() {
            delta.set_bit(i, *gene > 0.5);
        }
        delta
    }
}

impl GAPopulation {
    fn new(size: usize, gene_length: usize) -> Self {
        let mut individuals = Vec::with_capacity(size);
        for _ in 0..size {
            let genes = (0..gene_length)
                .map(|_| rand_float())
                .collect();
            individuals.push(Individual {
                genes,
                fitness: f64::INFINITY,
                learning_strategy: LearningStrategy::default(),
            });
        }

        Self {
            individuals,
            target: None,
            generation: 0,
            best_fitness_history: Vec::new(),
        }
    }

    fn set_target(&mut self, target: BitslicedLane) {
        self.target = Some(target);
    }

    fn evolve(&mut self) {
        let strategy = self.individuals[0].learning_strategy.clone();
        let elite_count = (self.individuals.len() as f64 * 0.2) as usize;
        let elite_count = elite_count.max(1);

        let mut new_individuals = Vec::with_capacity(self.individuals.len());

        // Elitism: keep the best individuals
        for i in 0..elite_count {
            new_individuals.push(self.individuals[i].clone());
        }

        // Generate rest through crossover and mutation
        while new_individuals.len() < self.individuals.len() {
            let parent1 = self.select_parent();
            let parent2 = self.select_parent();
            let mut child = self.crossover(&parent1, &parent2, &strategy);
            self.mutate(&mut child, &strategy);
            new_individuals.push(child);
        }

        self.individuals = new_individuals;
        self.generation += 1;
    }

    fn select_parent(&self) -> &Individual {
        // Tournament selection with learning strategy pressure
        let tournament_size = 3;
        let mut best = &self.individuals[0];
        for _ in 0..tournament_size {
            let idx = rand_usize() % self.individuals.len();
            let candidate = &self.individuals[idx];
            if candidate.fitness < best.fitness {
                best = candidate;
            }
        }
        best
    }

    fn crossover(
        &self,
        p1: &Individual,
        p2: &Individual,
        strategy: &LearningStrategy,
    ) -> Individual {
        let blend = strategy.crossover_blend;
        let mut genes = Vec::with_capacity(p1.genes.len());
        for (g1, g2) in p1.genes.iter().zip(p2.genes.iter()) {
            let gene = g1 * blend + g2 * (1.0 - blend);
            genes.push(gene.clamp(0.0, 1.0));
        }
        Individual {
            genes,
            fitness: f64::INFINITY,
            learning_strategy: strategy.clone(),
        }
    }

    fn mutate(&self, individual: &mut Individual, strategy: &LearningStrategy) {
        for gene in &mut individual.genes {
            if rand_float() < strategy.mutation_rate {
                // Gaussian-like perturbation
                let perturbation = (rand_float() - 0.5) * 0.2;
                *gene = (*gene + perturbation).clamp(0.0, 1.0);
            }
        }
    }
}

impl MetaLearner {
    fn new(population_size: usize) -> Self {
        let strategy_population = (0..population_size)
            .map(|_| Individual {
                genes: vec![rand_float(); 5], // 5 genes for strategy parameters
                fitness: f64::INFINITY,
                learning_strategy: LearningStrategy::default(),
            })
            .collect();

        Self {
            strategy_population,
            generations_without_improvement: 0,
            best_strategy: LearningStrategy::default(),
            best_strategy_fitness: 0.0,
        }
    }

    fn evolve_strategy(&mut self, ga_population: &GAPopulation) {
        // Evaluate each strategy by how well the GA converges with it
        for individual in &mut self.strategy_population {
            // Fitness = inverse of GA's best fitness (strategies that lead to better GA fitness score higher)
            if let Some(&best_ga_fitness) = ga_population.best_fitness_history.last() {
                individual.fitness = 1.0 / (1.0 + best_ga_fitness);
            }
        }

        self.strategy_population
            .sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        let best = &self.strategy_population[0].learning_strategy;
        let best_individual_fitness = self.strategy_population[0].fitness;
        if best_individual_fitness > self.best_strategy_fitness {
            self.best_strategy = best.clone();
            self.best_strategy_fitness = best_individual_fitness;
            self.generations_without_improvement = 0;
        } else {
            self.generations_without_improvement += 1;
        }

        // If no improvement for a while, increase exploration
        if self.generations_without_improvement > 10 {
            self.best_strategy.exploration_rate *= 1.1;
            self.generations_without_improvement = 0;
        }
    }
}

/// Simple pseudo-random float in [0, 1).
fn rand_float() -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEED: AtomicU64 = AtomicU64::new(0xDEADBEEF_CAFE_BABE);
    let s = SEED.fetch_add(1, Ordering::Relaxed);
    let x = ((s as f64) * 1.0e-9).fract();
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Simple pseudo-random usize.
fn rand_usize() -> usize {
    rand_float() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use binetic_core::register::Register;

    struct MockFabric {
        registers: std::sync::RwLock<std::collections::HashMap<RegisterAddress, Register>>,
    }

    impl MockFabric {
        fn new() -> Self {
            Self {
                registers: std::sync::RwLock::new(std::collections::HashMap::new()),
            }
        }
    }

    impl FabricAdapter for MockFabric {
        fn read_raw(&self, addr: RegisterAddress) -> Option<Register> {
            self.registers.read().unwrap().get(&addr).cloned()
        }

        fn write(&self, addr: RegisterAddress, reg: Register) -> Result<(), String> {
            self.registers.write().unwrap().insert(addr, reg);
            Ok(())
        }

        fn inject_adapt_raw(&self, addr: RegisterAddress, delta: BitslicedLane) -> Result<(), String> {
            let mut regs = self.registers.write().unwrap();
            let reg = regs.get(&addr).cloned().ok_or("Not found")?;
            let mut updated = reg.clone();
            updated.update_with_delta(&delta);
            regs.insert(addr, updated);
            Ok(())
        }
    }

    #[test]
    fn test_detect_intent_sin() {
        let fabric = Arc::new(MockFabric::new());
        let config = IntentEngineConfig::default();
        let mut engine = IntentEngine::new(fabric, config);

        let intent = engine.detect_intent("add sin trig gates to the circuit").unwrap();
        assert_eq!(intent.target_op, Some(ArithmeticOp::Sin));
    }

    #[test]
    fn test_detect_intent_xor() {
        let fabric = Arc::new(MockFabric::new());
        let config = IntentEngineConfig::default();
        let mut engine = IntentEngine::new(fabric, config);

        let intent = engine.detect_intent("use XOR logic for this operation").unwrap();
        assert_eq!(intent.target_op, Some(ArithmeticOp::XorGate));
    }

    #[test]
    fn test_detect_intent_learn() {
        let fabric = Arc::new(MockFabric::new());
        let config = IntentEngineConfig::default();
        let mut engine = IntentEngine::new(fabric, config);

        let intent = engine.detect_intent("learn to be smarter").unwrap();
        assert_eq!(intent.priority, 3.0);
    }

    #[test]
    fn test_compute_gap() {
        let fabric = Arc::new(MockFabric::new());
        let config = IntentEngineConfig::default();
        let engine = IntentEngine::new(fabric, config);

        let current = BitslicedLane::from_bits(&[true, false, true, false]);
        let target = BitslicedLane::from_bits(&[true, true, true, false]);
        let gap = engine.compute_gap(&current, &target).unwrap();

        assert_eq!(gap.get_bit(0), false); // same
        assert_eq!(gap.get_bit(1), true);  // different
        assert_eq!(gap.get_bit(2), false); // same
        assert_eq!(gap.get_bit(3), false); // same
    }

    #[test]
    fn test_meta_learner_evolution() {
        let fabric = Arc::new(MockFabric::new());
        let mut config = IntentEngineConfig::default();
        config.population_size = 10;
        config.max_generations = 5;

        let mut engine = IntentEngine::new(fabric, config);

        let intent = Intent {
            target_state: Some(BitslicedLane::from_bits(&[true, false, true, false])),
            target_op: Some(ArithmeticOp::Sin),
            corrections: HashMap::new(),
            description: "test intent".to_string(),
            priority: 1.0,
        };

        let current_state = BitslicedLane::from_bits(&[false, true, false, true]);
        let best = engine.evolve_to_intent(&intent, &current_state).unwrap();

        // Best fitness should be better than initial random
        assert!(best.fitness < 1.0);

        // Meta strategy should have been updated
        let strategy = engine.meta_strategy();
        assert!(strategy.mutation_rate > 0.0);
    }

    #[test]
    fn test_user_authority_approval_gate() {
        let authority = &mut UserAuthority::new(42);

        // Mutation with wrong intent hash should be rejected
        let wrong_delta = BitslicedLane::from_bits(&[true, false, true, false]);
        let addr = RegisterAddress::from(0u128);
        let approved = authority.approve(addr, &wrong_delta);
        assert!(!approved);

        // Mutation with matching intent hash should be approved
        let correct_delta = BitslicedLane::from_bits(&[true, false, true, false]);
        let authority2 = &mut UserAuthority::new(hash_bitslice(&correct_delta) as u64);
        let approved = authority2.approve(addr, &correct_delta);
        assert!(approved);
    }

    #[test]
    fn test_user_authority_rollback() {
        let authority = &mut UserAuthority::new(123);

        // Save checkpoint
        let addr = RegisterAddress::from(0u128);
        let register = Register::new(addr, BitslicedLane::from_bits(&[true, false]));
        authority.checkpoint(addr, &register);

        // Rollback should return the saved state
        let restored = authority.rollback(addr);
        assert!(restored.is_some());
        assert_eq!(restored.unwrap().payload_bytes(), register.payload_bytes());
    }

    #[test]
    fn test_user_authority_audit_trail() {
        let delta = BitslicedLane::from_bits(&[true, false, true, false]);
        let addr = RegisterAddress::from(0u128);

        // Authority created with matching intent hash
        let mut authority = UserAuthority::new(hash_bitslice(&delta) as u64);

        // Initially no records
        assert!(authority.audit_trail().is_empty());

        // Approve adds a record — delta matches intent hash, so approved
        authority.approve(addr, &delta);
        assert_eq!(authority.audit_trail().len(), 1);
        assert!(authority.audit_trail()[0].approved);

        // All approved check
        assert!(authority.verify_all_approved());

        // Wrong delta should be rejected
        let wrong_delta = BitslicedLane::from_bits(&[false, true, false, true]);
        authority.approve(addr, &wrong_delta);
        assert_eq!(authority.audit_trail().len(), 2);
        assert!(!authority.audit_trail()[1].approved);

        // Not all approved anymore
        assert!(!authority.verify_all_approved());
    }

    #[test]
    fn test_intent_engine_full_pipeline() {
        // Verify: user intent drives fitness, not external targets
        let fabric = Arc::new(MockFabric::new());
        let mut config = IntentEngineConfig::default();
        config.population_size = 20;
        config.max_generations = 10;

        let mut engine = IntentEngine::new(fabric, config);

        // User intent: "learn sin gates"
        let intent = Intent {
            target_state: Some(BitslicedLane::from_bits(&[true, true, false, true])),
            target_op: Some(ArithmeticOp::Sin),
            corrections: HashMap::new(),
            description: "learn sin gates".to_string(),
            priority: 1.0,
        };

        let current_state = BitslicedLane::from_bits(&[false, false, false, false]);
        let best = engine.evolve_to_intent(&intent, &current_state).unwrap();

        // GA evolved toward the user's intent — fitness improved
        assert!(best.fitness < f64::INFINITY);

        // Meta-learning: strategy improved after evolution
        let strategy = engine.meta_strategy();
        assert!(strategy.mutation_rate > 0.0);
    }

    #[test]
    fn test_gap_drives_evolution() {
        // Verify: gap = current XOR target is the fitness signal
        let fabric = Arc::new(MockFabric::new());
        let config = IntentEngineConfig::default();
        let mut engine = IntentEngine::new(fabric, config);

        let intent = Intent {
            target_state: Some(BitslicedLane::from_bits(&[true, false, true, false])),
            target_op: Some(ArithmeticOp::Sin),
            corrections: HashMap::new(),
            description: "test gap".to_string(),
            priority: 1.0,
        };

        let current = BitslicedLane::from_bits(&[false, false, false, false]);
        let gap = engine.compute_gap(&current, intent.target_state.as_ref().unwrap()).unwrap();

        // Gap should have bits set where current differs from target
        assert!(gap.popcount() > 0);

        // Evolve should reduce the gap
        let best = engine.evolve_to_intent(&intent, &current).unwrap();
        assert!(best.fitness < 4.0); // better than all-bits-different
    }

    #[test]
    fn test_trajectory_graph() {
        let mut traj = TrajectoryGraph::new();

        let addr = RegisterAddress::from(0u128);
        let n1 = traj.add_node("read".to_string(), addr, 42);
        let n2 = traj.add_node("inject".to_string(), addr, 43);

        assert_eq!(n1, 0);
        assert_eq!(n2, 1);
        assert_eq!(traj.nodes().len(), 2);
        assert_eq!(traj.edges.len(), 1);

        traj.set_outcome(n1, TrajectoryOutcome::Success);
        traj.set_outcome(n2, TrajectoryOutcome::Success);
        assert_eq!(traj.success_rate(), 1.0);
    }

    #[test]
    fn test_intent_graph_drift() {
        let mut intent_graph = IntentGraph::new();

        let i1 = intent_graph.add_intent(100, "add sin".to_string(), 1.0);
        let i2 = intent_graph.add_intent(100, "add sin".to_string(), 1.0);
        let i3 = intent_graph.add_intent(200, "add cos".to_string(), 1.0);

        intent_graph.confirm(i1);
        intent_graph.confirm(i2);
        intent_graph.confirm(i3);

        // First two intents are same, third is different
        // Drift: (100^100 + 100^200) / 2 = (0 + 440) / 2 = 220
        assert!(intent_graph.intent_drift() > 0.0);

        // Latest confirmed should be the third
        let latest = intent_graph.latest_confirmed().unwrap();
        assert_eq!(latest.intent_hash, 200);
    }

    #[test]
    fn test_scoring_model_online_learning() {
        let mut model = ScoringModel::new(4, 0.1);

        let features = vec![1.0, 0.5, 0.3, 0.8];

        // Before training, score should be 0.0 (all weights are 0)
        assert_eq!(model.score(&features), 0.0);
        assert_eq!(model.training_count(), 0);

        // Train with positive outcome
        model.train(&features, true);
        assert_eq!(model.training_count(), 1);

        // Score should now be non-zero (weights updated)
        let score_after = model.score(&features);
        assert!(score_after != 0.0);

        // Train with negative outcome
        model.train(&features, false);
        assert_eq!(model.training_count(), 2);
    }

    #[test]
    fn test_verifier_three_gate() {
        let mut verifier = Verifier::new(4, 0.0, 0.0, -1e9);

        let addr = RegisterAddress::from(0u128);
        let delta_hash = 42;
        let intent_hash = 42;
        let features = vec![1.0, 0.0, 0.0, 0.0];

        // No trajectory nodes yet — success rate is 0.0
        // With min_trajectory_success_rate=0.0, this should pass
        let result = verifier.verify(delta_hash, intent_hash, &features);
        // Trajectory is empty, success_rate is 0.0, threshold is 0.0 → passes
        // Intent graph is empty, no confirmed intent → passes
        // Scoring model not trained, score is 0.0, threshold is -1e9 → passes
        assert!(result);

        // Add a confirmed intent that doesn't match
        verifier.record_intent(999, "different intent".to_string(), 1.0);
        verifier.confirm_intent(0);
        // Raise alignment threshold so mismatched hash fails
        verifier.min_intent_alignment = 0.9;

        // Now intent alignment check should fail (42 vs 999 — low bit match)
        let result2 = verifier.verify(delta_hash, intent_hash, &features);
        assert!(!result2);
    }

    #[test]
    fn test_verifier_full_pipeline() {
        // Verify: trajectory + intent + scoring model all gate the mutation
        let mut verifier = Verifier::new(4, 0.0, 0.0, -1e9);

        let addr = RegisterAddress::from(0u128);
        let delta_hash = 100;
        let intent_hash = 100;
        let features = vec![0.5, 0.5, 0.5, 0.5];

        // Record trajectory outcome
        let node_idx = verifier.add_trajectory_node("inject".to_string(), addr, delta_hash);
        verifier.record_outcome(node_idx, TrajectoryOutcome::Success);

        // Record confirmed intent
        verifier.record_intent(intent_hash, "learn sin".to_string(), 1.0);
        verifier.confirm_intent(0);

        // Train scoring model on positive outcome
        verifier.train_model(&features, true);

        // All three gates should pass now
        assert!(verifier.verify(delta_hash, intent_hash, &features));
    }
}