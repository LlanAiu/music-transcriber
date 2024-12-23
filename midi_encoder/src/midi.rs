// builtin

// external
use midly::{Smf, Timing, TrackEventKind};

// internal
use crate::types::NoteEvent;


pub fn parse_midi(file_path: &str) -> Vec<NoteEvent> {
    let data = std::fs::read(file_path).expect("Failed to read MIDI file");
    let smf = Smf::parse(&data).expect("Failed to parse MIDI file");

    let tpq = match smf.header.timing {
        Timing::Metrical(ticks_per_beat) => ticks_per_beat.as_int() as u32,
        _ => panic!("Timing not in ticks per beat"),
    };

    let mut tempo = 500_000; //in microseconds per beat
    let mut current_ticks: u32 = 0;
    let mut events = Vec::new();

    for track in smf.tracks {
        for event in track {
            current_ticks += event.delta.as_int() as u32;

            if let TrackEventKind::Meta(midly::MetaMessage::Tempo(new_tempo)) = event.kind {
                tempo = new_tempo.as_int();
            }

            if let TrackEventKind::Midi {
                channel: _,
                message,
            } = event.kind
            {
                let timestamp_ms =
                    (current_ticks as f64 * tempo as f64 / tpq as f64 / 1_000.0) as u32;
                match message {
                    midly::MidiMessage::NoteOn { key, vel: _ } => {
                        events.push(NoteEvent::new(timestamp_ms, key.as_int(), true));
                    }
                    midly::MidiMessage::NoteOff { key, vel: _ } => {
                        events.push(NoteEvent::new(timestamp_ms, key.as_int(), false));
                    }
                    _ => {}
                }
            }
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_midi() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");
        println!("{:?}", events);
        assert!(!events.is_empty());
    }
}