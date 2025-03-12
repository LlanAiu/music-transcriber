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

    let mut last_time_ms: f32 = 0.0;
    let mut time_ms: f32 = 0.0;

    for track in smf.tracks {
        for event in track {
            time_ms += ticks_to_ms(event.delta.as_int() as u32, tempo as f64, tpq as f64);

            if let TrackEventKind::Meta(midly::MetaMessage::Tempo(new_tempo)) = event.kind {
                tempo = new_tempo.as_int();
            }

            if let TrackEventKind::Midi {
                channel: _,
                message,
            } = event.kind
            {
                match message {
                    midly::MidiMessage::NoteOn { key, vel } => {
                        let time_delta = ((time_ms - last_time_ms) / 5.0).round() * 5.0;
                        last_time_ms = time_ms;
                        if vel.as_int() == 0 {
                            events.push(NoteEvent::new(time_delta, key.as_int(), false));
                        } else {
                            events.push(NoteEvent::new(time_delta, key.as_int(), true));
                        }
                    }
                    midly::MidiMessage::NoteOff { key, vel: _ } => {
                        let time_delta = ((time_ms - last_time_ms) / 5.0).round() * 5.0;
                        last_time_ms = time_ms;
                        events.push(NoteEvent::new(time_delta, key.as_int(), false));
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

    let tempo: u32 = 500_000; // microseconds
    let tpq: u16 = 480;

    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(tempo))),
    });

    for event in events {
        let tick_delta: u32 = ms_to_ticks(event.get_time_delta(), tempo as f64, tpq as f64);

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

fn ticks_to_ms(ticks: u32, tempo: f64, tpq: f64) -> f32 {
    (ticks as f64 * tempo / tpq / 1_000.0) as f32
}

fn ms_to_ticks(ms: f32, tempo: f64, tpq: f64) -> u32 {
    (ms as f64 * tpq * 1_000.0 / tempo) as u32
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
        let events_head: Vec<NoteEvent> = events.iter().take(100).cloned().collect();
        println!("{:#?}", events_head);
        assert!(!events.is_empty());
    }

    #[test]
    fn written_data_test() {
        let events: Vec<NoteEvent> = parse_midi("./tests/output/Data_test.mid");
        let events_head: Vec<NoteEvent> = events.iter().take(100).cloned().collect();
        println!("{:#?}", events_head);
        assert!(!events.is_empty());
    }

    #[test]
    fn test_write_midi() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");
        write_midi(&events, "./tests/output/C_only.mid");
    }

    #[test]
    fn test_write_midi_complex() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Double_Note_Test.mid");
        write_midi(&events, "./tests/output/Double_note.mid");
    }

    #[test]
    fn test_write_midi_data() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Data_Test.midi");
        write_midi(&events, "./tests/output/Data_test.mid");
    }
}
