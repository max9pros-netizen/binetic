use temporal_packet_network::engine::SimulationEngine;
use temporal_packet_network::gate::TemporalGate;
use temporal_packet_network::packet::Packet;
use temporal_packet_network::primitives::{Rail, Route, TemporalWindow};

#[test]
fn test_timing_window_on_time_and_miss() {
    let mut engine = SimulationEngine::new(0);
    let gate = TemporalGate::new(
        0x100,
        TemporalWindow::new(50, 100),
        Route::new(vec![]),
        Route::new(vec![]),
    );
    engine.register_gate(gate);

    let pkt_on_time = Packet::new([0u8; 16], Rail::A, 0, 0, 200, [0u8; 32], [0u8; 32], vec![]);
    let pkt_late = Packet::new([0u8; 16], Rail::A, 1, 0, 80, [0u8; 32], [0u8; 32], vec![]);

    engine.schedule_packet(0x100, 75, pkt_on_time);
    engine.schedule_packet(0x100, 150, pkt_late);

    engine.run_until_idle().unwrap();

    assert!(engine.execution_log.iter().any(|l| l.contains("OnTime")));
    assert!(engine.execution_log.iter().any(|l| l.contains("Missed")));
}
