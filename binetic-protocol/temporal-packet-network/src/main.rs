use temporal_packet_network::crypto::xor_hashes;
use temporal_packet_network::engine::SimulationEngine;
use temporal_packet_network::gate::TemporalGate;
use temporal_packet_network::packet::Packet;
use temporal_packet_network::primitives::{Rail, Route, TemporalWindow};
use temporal_packet_network::utm::{Symbol, Utm2State3Symbol};

fn main() {
    println!("=== Temporal Packet Network (TPN) Discrete-Event Engine ===");

    let mut engine = SimulationEngine::new(1000);
    let secret_key = [0x42u8; 32];

    let gate_1 = TemporalGate::new(
        0x1000,
        TemporalWindow::new(1000, 1050),
        Route::new(vec![]),
        Route::new(vec![]),
    );

    let gate_2 = TemporalGate::new(
        0x2000,
        TemporalWindow::new(1020, 1080),
        Route::new(vec![]),
        Route::new(vec![]),
    ).with_ticket(secret_key);

    engine.register_gate(gate_1);
    engine.register_gate(gate_2);

    let pkt_a = Packet::new([1u8; 16], Rail::A, 0, 1000, 1200, [0u8; 32], [0u8; 32], vec![1, 2, 3]);
    // The ticket that gate_2 requires is xor(secret, control_hash): when Rail A
    // passes gate_1, the engine sets its control register to pkt_a's chain_hash.
    let ticket = xor_hashes(&secret_key, &pkt_a.header.chain_hash);
    let mut pkt_b = Packet::new([1u8; 16], Rail::B, 0, 1000, 1200, [0u8; 32], [0u8; 32], vec![4, 5, 6]);
    pkt_b = pkt_b.with_ticket(ticket);

    engine.schedule_packet(0x1000, 1025, pkt_a);
    engine.schedule_packet(0x2000, 1040, pkt_b);

    println!("\n[Running Simulation Engine...]");
    engine.run_until_idle().expect("Simulation failed");

    for log in &engine.execution_log {
        println!("{}", log);
    }

    println!("\n=== Running UTM (2-State 3-Symbol) ===");
    let mut utm = Utm2State3Symbol::new();
    utm.load_tape(&[(0, Symbol::Zero), (1, Symbol::One)]);

    while utm.step() {
        if utm.step_count == 1
            || utm.step_count % 1000 == 0
            || utm.head_pos == 1
            || utm.head_pos == 334
            || utm.head_pos == 6666667
        {
            println!(
                "Step {}: State={:?}, Head={}",
                utm.step_count, utm.state, utm.head_pos
            );
        }
    }
    println!("UTM Finished in {} steps with final state {:?}", utm.step_count, utm.state);
}
