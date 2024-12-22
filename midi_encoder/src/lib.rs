use core::panic;
use std::vec;

use midly::{Smf, Timing, TrackEventKind};


const ENCODING_LENGTH: usize = 88 * 2;

#[derive(Debug)]
pub struct NoteEvent {
    timestamp: u32, // in ms
    key: u8,
    note_on: bool,
}

pub struct MIDIEncoding {
    encoding: Vec<Vec<f32>>,
}


impl NoteEvent {
    pub fn new(timestamp: u32, key: u8, note_on: bool) -> NoteEvent {
        NoteEvent {
            timestamp,
            key,
            note_on,
        }
    }
}

pub fn parse_midi(file_path: &str) -> Vec<NoteEvent> {
    let data = std::fs::read(file_path).expect("Failed to read MIDI file");
    let smf = Smf::parse(&data).expect("Failed to parse MIDI file");

    let tpq = match smf.header.timing {
        Timing::Metrical(ticks_per_beat) => ticks_per_beat.as_int() as u32,
        _ => panic!("Timing not in ticks per beat"),
    };

    let mut tempo = 500_000; //microseconds per beat, default MIDI value
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

pub fn encode_midi(events: Vec<NoteEvent>, timestep_ms: u32) -> MIDIEncoding {

    let mut current_time: u32 = 0;

    let mut encoding: Vec<Vec<f32>> = Vec::new();
    let mut previous_note: Option<Vec<f32>> = None;

    for event in events {
        println!("Processing: {:?}", event);
        if previous_note.is_some() && event.timestamp.abs_diff(current_time) < timestep_ms {
            let prev: Vec<f32> = previous_note.expect("Cannot layer null note encoding");
            let note: Vec<f32> = get_note_encoding(event).expect("Failed to get note encoding");
            previous_note = layer(&prev, &note);
        } else {
            if previous_note.is_some() {
                let note: Vec<f32> = previous_note.expect("Cannot get note encoding");
                encoding.push(note);
                current_time += timestep_ms;
            }
            let next_time = event.timestamp;
            while current_time + timestep_ms <= next_time {
                current_time += timestep_ms;
                encoding.push(vec![0.0; ENCODING_LENGTH]);
            }
            previous_note = get_note_encoding(event);
        } 
    }
    encoding.push(previous_note.expect("Cannot get note encoding"));

    MIDIEncoding { encoding: encoding }
}

/* 
 MIDI values go from [21, 108]
 indices 2x = on/off, 2x + 1 for note ID 
*/
fn get_note_encoding(note: NoteEvent) -> Option<Vec<f32>> {
    let mut note_encoding: Vec<f32> = vec![0.0; ENCODING_LENGTH];
    let note_index: usize = (2 * (note.key - 21) + 1) as usize;
    note_encoding[note_index] = 1.0;
    if note.note_on {
        note_encoding[note_index - 1] = 1.0;
    }
    Some(note_encoding)
}

fn decode_note_encoding(encoding: Vec<f32>, timestamp_ms: u32) -> Option<NoteEvent>{
    if encoding.len() != ENCODING_LENGTH {
        panic!("Not a valid note encoding");
    }

    for i in (1..encoding.len()).skip(1).step_by(2) {
        if encoding[i] == 1.0 {
            let key: u8 = ((i - 1) / 2 + 21) as u8;
            let event: NoteEvent = NoteEvent::new(timestamp_ms, key, encoding[i - 1] == 1.0);
            return Option::Some(event);
        }
    }

    Option::None
}

fn layer(n1: &Vec<f32>, n2: &Vec<f32>) -> Option<Vec<f32>> {
    if n1.len() != n2.len() {
        panic!("Cannot add vectors of differing length");
    }
    let mut new_encoding: Vec<f32> = Vec::new();
    for i in 0..n1.len() {
        new_encoding.push(n1[i] + n2[i]);
    }

    Some(new_encoding)
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
    fn encode_simple_midi() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");
        let encoding: MIDIEncoding = encode_midi(events, 500);
        println!("{}", encoding.encoding.len());
        for (i, vector) in encoding.encoding.iter().enumerate() {
            println!("Timestep {i}: {:?}", vector);
        }
    }

    #[test]
    fn encode_complex_midi() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Double_Note_Test.mid");
        let encoding: MIDIEncoding = encode_midi(events, 250);
        println!("{}", encoding.encoding.len());
        for (i, vector) in encoding.encoding.iter().enumerate() {
            println!("Timestep {i}: {:?}", vector);
        }
    }

    #[test]
    fn layer_test(){
        let n1: Vec<f32> = vec![1.0; 5];
        let n2: Vec<f32> = vec![2.0; 5];
        assert_eq!(layer(&n1, &n2), Some(vec![3.0; 5]));
    }
}
