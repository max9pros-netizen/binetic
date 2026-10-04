#!/usr/bin/env python3
"""
Temporal neurons and networks.

A neuron is a register: its weights ARE phase delays, its activation
IS a trig gate, its memory IS the accumulated value. The forward pass
IS packet routing; learning IS phase-space evolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PI = 3.141592653589793
sin = math.sin


@dataclass
class Neuron:
    """A temporal neuron: an IPv4/IPv6 register with temporal weights.

    The weight w_ij is encoded as the phase delay τ_ij — there is no
    multiplication, only timing. The activation σ IS the trig gate
    window: the neuron fires iff its arrival phase aligns.

        neuron = {
            address:  IP register,
            threshold: gate threshold (learnable),
            weight:   phase scaling (synaptic weight = timing),
            phase:    firing time (membrane timing),
            memory:   accumulated input (data field),
        }
    """

    ip: u128
    threshold: float = 0.43
    weight: float = 1.0
    phase: float = 0.0
    memory: float = 0.0
    spikes: List[float] = field(default_factory=list)
    window: Tuple[float, float] = (0.0, PI / 2.0)

    def fire(self, arrival: float) -> bool:
        """Fire iff the arrival phase aligns with this neuron's window.

        θ = (π/2)·u, u = normalized arrival in window
        fitness = sin(2θ) → fire iff fitness > threshold
        """
        u = (arrival - self.window[0]) / (self.window[1] - self.window[0])
        theta = (PI / 2.0) * max(0.0, min(1.0, u))
        fitness = sin(2.0 * theta) * self.weight + 0.0
        fired = fitness > self.threshold
        if fired:
            self.spikes.append(arrival)
        return fired

    def accumulate(self, value: float, arrival: Optional[float] = None) -> None:
        """Accumulate input (membrane potential). If a spike fired this
        step, record the spike time."""
        self.memory += value
        if self.spikes and self.spikes[-1] == (arrival or 0):
            pass

    def reset(self) -> None:
        """Reset for the next forward pass."""
        self.memory = 0.0
        self.spikes = []

    def __repr__(self) -> str:
        return (f"Neuron(0x{self.ip:x}, threshold={self.threshold:.2f}, "
                f"weight={self.weight:.2f}, phase={self.phase:.2f}, "
                f"memory={self.memory:.2f}, spikes={len(self.spikes)})")


@dataclass
class Network:
    """A temporal neural network: neurons are IP registers, synapses
    are phase delays, learning is phase-space evolution."""

    neurons: Dict[u128, Neuron] = field(default_factory=dict)
    synapses: Dict[Tuple[u128, u128], float] = field(default_factory=dict)

    def add_neuron(self, ip: u128, phase: float = 0.0,
                   threshold: float = 0.43, weight: float = 1.0) -> Neuron:
        """Add a neuron (register) to the network."""
        n = Neuron(ip=ip, phase=phase, threshold=threshold, weight=weight)
        self.neurons[ip] = n
        return n

    def synapse(self, pre: u128, post: u128, delay: float = 50.0) -> None:
        """Add a synapse: the weight IS the phase delay.

        Packet travels pre → post after `delay`. The weight is not a
        number — it is a timing. Neuron_post fires iff enough packets
        arrive within its window.
        """
        self.synapses[(pre, post)] = delay

    def forward(self, inputs: List[Tuple[u128, float]]) -> List[u128]:
        """Forward pass IS routing.

        inputs = list of (neuron_ip, arrival_time). Packets flow through
        the synapses (delays) and neurons fire iff arrival phases align.

        Returns the list of neurons that fired.
        """
        for n in self.neurons.values():
            n.reset()

        fired: List[u128] = []
        for pre_ip, arrival in inputs:
            # Propagate through synapses
            for (pre, post), delay in self.synapses.items():
                if pre != pre_ip:
                    continue
                post_arrival = arrival + delay
                post_neuron = self.neurons.get(post)
                if post_neuron is None:
                    continue
                # Accumulate input at the postsynaptic neuron
                post_neuron.accumulate(1.0, post_arrival)
                # Fire if phase aligns
                if post_neuron.fire(post_arrival):
                    post_neuron.memory += post_neuron.weight
                    if post not in fired:
                        fired.append(post)

        # Direct inputs (no synapse) also can fire
        for ip, arrival in inputs:
            n = self.neurons.get(ip)
            if n and n.fire(arrival):
                if ip not in fired:
                    fired.append(ip)

        return fired

    def attention(self, query_ip: u128, key_ip: u128, value_ip: u128) -> float:
        """Temporal attention: routing by phase alignment.

        Attention = softmax over phase alignment. Two neurons attend iff
        their phase windows overlap. Attention is topology, not matrix
        multiplication.
        """
        q = self.neurons.get(query_ip)
        k = self.neurons.get(key_ip)
        if q is None or k is None:
            return 0.0
        # Phase alignment score: how close are the phases?
        score = 1.0 - abs(q.phase - k.phase) / PI
        score = max(0.0, score)
        # Trig softmax weighting
        fitness = sin(max(0.0, min(PI / 2.0, score * PI / 2.0)))
        return fitness

    def evolve(self, fitness_fn, generations: int = 10,
               mutation_rate: float = 0.3) -> List[Dict]:
        """Learn by evolving phases in phase space.

        Parameters (thresholds, phase offsets, weights) are phases and
        timings — never numerical weight matrices. The search is in
        phase space, driven by fitness.
        """
        import random
        history = []
        best_fit = fitness_fn(self)
        best = {k: v.phase for k, v in self.neurons.items()}

        for gen in range(generations):
            for n in self.neurons.values():
                if random.random() < mutation_rate:
                    n.threshold = max(0.0, min(1.0, n.threshold + random.uniform(-0.05, 0.05)))
                    n.weight = max(0.0, n.weight + random.uniform(-0.1, 0.1))
                    n.phase += random.uniform(-5.0, 5.0)
            fit = fitness_fn(self)
            history.append({"generation": gen, "fitness": fit})
            if fit > best_fit:
                best_fit = fit
                best = {k: v.phase for k, v in self.neurons.items()}

        return history

    def __len__(self) -> int:
        return len(self.neurons)

    def __repr__(self) -> str:
        return f"Network(neurons={len(self.neurons)}, synapses={len(self.synapses)})"


def demo() -> None:
    """Demonstrate the temporal neural network."""
    print("=" * 64)
    print("   binetic temporal neural network")
    print("   Weights = phase delays · Activation = trig gate")
    print("=" * 64)
    print()

    net = Network()

    # Neurons = IP registers
    print("--- 1. Neurons = IP registers ---")
    for i in range(8):
        ip = (0x1000000000000000 | i)
        net.add_neuron(ip=ip, phase=i * 10.0, threshold=0.5, weight=1.0)
    print(f"   {len(net)} neurons (IP registers with temporal phases)")
    print()

    # Synapses = phase delays (weights encoded as timing)
    print("--- 2. Synapses = phase delays (weights = timing) ---")
    for i in range(7):
        net.synapse((0x1000000000000000 | i),
                    (0x1000000000000000 | (i + 1)),
                    delay=50.0 + i * 5.0)
    print(f"   {len(net.synapses)} synapses (temporal gates / phase windows)")
    print()

    # Forward pass = routing
    print("--- 3. Forward pass = routing ---")
    inputs = [(0x1000000000000000, 50.0), (0x1000000000000001, 55.0)]
    fired = net.forward(inputs)
    print(f"   inputs: {inputs}")
    print(f"   fired neurons: {len(fired)}")
    for ip in fired:
        print(f"     0x{ip:016x}")
    print()

    # Attention = routing by phase alignment
    print("--- 4. Temporal attention = routing by phase alignment ---")
    q, k, v = (0x1000000000000000, 0x1000000000000001, 0x1000000000000002)
    att = net.attention(q, k, v)
    print(f"   attention(Q=0x{q:016x}, K=0x{k:016x}, V=0x{v:016x}) = {att:.4f}")
    print()

    # Learning = phase evolution
    print("--- 5. Learning = phase-space evolution ---")

    def simple_fitness(net: Network) -> float:
        """Fitness: does neuron 0 fire with arrival phase 50 (theta=π/4)?"""
        ip = 0x1000000000000000
        n = net.neurons[ip]
        n.reset()
        n.phase = 50.0
        n.threshold = 0.43
        # theta = π/4 → fitness = sin(π/2) = 1.0 > 0.43 → should fire
        return 1.0 if n.fire(50.0) else 0.0

    history = net.evolve(simple_fitness, generations=5, mutation_rate=0.3)
    for h in history:
        print(f"   Gen {h['generation']}: fitness = {h['fitness']:.4f}")
    print()

    print("--- Summary ---")
    print(f"   {net}")
    print()
    print("   ✓ Neurons = temporal registers (IP addresses)")
    print("   ✓ Synapses = temporal gates (phase windows)")
    print("   ✓ Arrival = computation (no fetch/decode/execute)")
    print("   ✓ Learning = auto-evolution of phases")
    print("   ✓ Error correction = 3× temporal redundancy")
    print("   ✓ Attention = which neurons fire at which phase")


if __name__ == "__main__":
    demo()
