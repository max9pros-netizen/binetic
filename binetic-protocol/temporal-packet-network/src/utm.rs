use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum State {
    Q0,
    Q1,
    Halt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Symbol {
    Blank,
    Zero,
    One,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Left,
    Right,
}

#[derive(Debug, Clone)]
pub struct Transition {
    pub write: Symbol,
    pub move_dir: Direction,
    pub next_state: State,
}

pub struct Utm2State3Symbol {
    pub tape: HashMap<i64, Symbol>,
    pub head_pos: i64,
    pub state: State,
    pub transitions: HashMap<(State, Symbol), Transition>,
    pub step_count: u64,
}

impl Utm2State3Symbol {
    pub fn new() -> Self {
        let mut transitions = HashMap::new();
        // Q0 Transitions
        transitions.insert((State::Q0, Symbol::Blank), Transition { write: Symbol::One, move_dir: Direction::Right, next_state: State::Q1 });
        transitions.insert((State::Q0, Symbol::Zero), Transition { write: Symbol::One, move_dir: Direction::Right, next_state: State::Q0 });
        transitions.insert((State::Q0, Symbol::One), Transition { write: Symbol::Zero, move_dir: Direction::Left, next_state: State::Q1 });
        // Q1 Transitions
        transitions.insert((State::Q1, Symbol::Blank), Transition { write: Symbol::Zero, move_dir: Direction::Left, next_state: State::Q0 });
        transitions.insert((State::Q1, Symbol::Zero), Transition { write: Symbol::One, move_dir: Direction::Left, next_state: State::Halt });
        transitions.insert((State::Q1, Symbol::One), Transition { write: Symbol::Blank, move_dir: Direction::Right, next_state: State::Q0 });

        Self {
            tape: HashMap::new(),
            head_pos: 0,
            state: State::Q0,
            transitions,
            step_count: 0,
        }
    }

    pub fn load_tape(&mut self, symbols: &[(i64, Symbol)]) {
        for (pos, sym) in symbols {
            self.tape.insert(*pos, *sym);
        }
    }

    pub fn step(&mut self) -> bool {
        if self.state == State::Halt {
            return false;
        }
        let current_symbol = *self.tape.get(&self.head_pos).unwrap_or(&Symbol::Blank);
        if let Some(trans) = self.transitions.get(&(self.state, current_symbol)).cloned() {
            self.tape.insert(self.head_pos, trans.write);
            match trans.move_dir {
                Direction::Left => self.head_pos -= 1,
                Direction::Right => self.head_pos += 1,
            }
            self.state = trans.next_state;
            self.step_count += 1;
            true
        } else {
            self.state = State::Halt;
            false
        }
    }
}
