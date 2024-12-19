use midly::{Smf, TrackEventKind};

pub struct NoteEvent {
    tick_delta: u32,
    key: u8,
    note_on: bool
}

impl NoteEvent {
    pub fn new(tick_delta: u32, key: u8, note_on: bool) -> NoteEvent {
        NoteEvent {tick_delta, key, note_on}
    }
}

pub fn parse_midi(file_path: &str) -> Vec<NoteEvent> {
    let data = std::fs::read(file_path).expect("Failed to read MIDI file");
    let smf = Smf::parse(&data).expect("Failed to parse MIDI file");

    let mut events = Vec::new();
    for track in smf.tracks {
        for event in track {
            if let TrackEventKind::Midi { channel: _, message } = event.kind {
                match message {
                    midly::MidiMessage::NoteOn { key, vel: _ } => {
                        events.push(
                            NoteEvent::new(
                                event.delta.as_int(),
                                key.as_int(),
                                true)
                        );
                    }
                    midly::MidiMessage::NoteOff { key, vel: _ } => {
                        events.push(
                            NoteEvent::new(
                                event.delta.as_int(),
                                key.as_int(),
                                false)
                        );
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
        let events = parse_midi("path/to/midi/file.mid");
        assert!(!events.is_empty());
    }
}
