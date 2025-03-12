// builtin

// external
use midly::{
    num::*, Header, MetaMessage, MidiMessage, Smf, Timing, Track, TrackEvent, TrackEventKind,
};

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
    let mut events = Vec::new();

    for track in smf.tracks {
        for event in track {
            let tick_delta: u32 = event.delta.as_int() as u32;

            if let TrackEventKind::Meta(midly::MetaMessage::Tempo(new_tempo)) = event.kind {
                tempo = new_tempo.as_int();
            }

            if let TrackEventKind::Midi {
                channel: _,
                message,
            } = event.kind
            {
                let time_delta_s: f32 =
                    (tick_delta as f64 * tempo as f64 / tpq as f64 / 1_000_000.0) as f32;

                match message {
                    midly::MidiMessage::NoteOn { key, vel } => {
                        if vel.as_int() == 0 {
                            events.push(NoteEvent::new(time_delta_s, key.as_int(), false));
                        } else {
                            events.push(NoteEvent::new(time_delta_s, key.as_int(), true));
                        }
                    }
                    midly::MidiMessage::NoteOff { key, vel: _ } => {
                        events.push(NoteEvent::new(time_delta_s, key.as_int(), false));
                    }
                    _ => {}
                }
            }
        }
    }

    events
}

pub fn write_midi(events: &Vec<NoteEvent>, file_path: &str) {
    let mut track = Track::new();

    let tempo = 500_000; // microseconds
    let tpq = 480;

    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(tempo))),
    });

    for event in events {
        let tick_delta: u32 =
            (event.get_time_delta() as f64 * tpq as f64 * 1_000_000.0 / tempo as f64) as u32;

        let delta: u28 = u28::from(tick_delta);
        let kind = if event.get_note_ref().is_note_on() {
            TrackEventKind::Midi {
                channel: u4::new(0),
                message: MidiMessage::NoteOn {
                    key: u7::new(event.get_note_ref().get_key()),
                    vel: u7::new(64),
                },
            }
        } else {
            TrackEventKind::Midi {
                channel: u4::new(0),
                message: MidiMessage::NoteOff {
                    key: u7::new(event.get_note_ref().get_key()),
                    vel: u7::new(64),
                },
            }
        };

        track.push(TrackEvent { delta, kind });
    }

    let smf = Smf {
        header: Header::new(
            midly::Format::SingleTrack,
            midly::Timing::Metrical(u15::new(tpq)),
        ),
        tracks: vec![track],
    };

    let mut buffer = Vec::new();
    smf.write(&mut buffer).expect("Failed to write MIDI data");

    std::fs::write(file_path, buffer).expect("Failed to write MIDI file");
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

    #[test]
    fn training_data_test() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Data_Test.midi");
        let events_head: Vec<NoteEvent> = events.iter().take(20).cloned().collect();
        println!("{:#?}", events_head);
        assert!(!events.is_empty());
    }

    #[test]
    fn test_write_midi() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");
        write_midi(&events, "./tests/output/C_only.mid");
    }
}
