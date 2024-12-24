// builtin

// external

// internal
use crate::types::{MIDIEncoding, NoteEvent, EncodingData};
use crate::constants::ENCODING_LENGTH;


//rethink due to too close note diffs?
pub fn encode(data: EncodingData) -> MIDIEncoding {
    let events: &Vec<NoteEvent> = data.get_events();
    let timestep_ms: u32 = data.get_timestep();
    
    let mut current_time: u32 = 0;
    let mut encoding: Vec<Vec<f32>> = Vec::new();
    let mut previous_note: Option<Vec<f32>> = None;

    for event in events.iter() {
        let should_layer: bool = previous_note.is_some()
            && event.get_timestamp().abs_diff(current_time) < timestep_ms;
        if should_layer {
            let prev: Vec<f32> = previous_note.expect("Cannot layer null note encoding");
            let note: Vec<f32> = get_note_encoding(event).expect("Failed to get note encoding");
            previous_note = layer_notes(&prev, &note);
        } else {
            if previous_note.is_some() {
                let note: Vec<f32> = previous_note.expect("Cannot get note encoding");
                encoding.push(note);
                current_time += timestep_ms;
            }
            let next_time = event.get_timestamp();
            while current_time + timestep_ms <= next_time {
                current_time += timestep_ms;
                encoding.push(vec![0.0; ENCODING_LENGTH]);
            }
            previous_note = get_note_encoding(event);
        }
    }
    encoding.push(previous_note.expect("Cannot get note encoding"));

    MIDIEncoding::new(timestep_ms, encoding)
}

pub fn decode(midi: MIDIEncoding) -> Vec<NoteEvent> {
    let mut events: Vec<NoteEvent> = Vec::new();

    for (i, encoding) in midi.get_encoding().iter().enumerate() {
        let decoded: Option<Vec<NoteEvent>> =
            decode_note_encoding(encoding, (i as u32) * midi.get_timestep());
        if decoded.is_some() {
            let event: Vec<NoteEvent> = decoded.expect("Failed to get decoded note event");
            for e in event {
                events.push(e);
            }
        }
    }

    events
}

/*
 MIDI values go from [21, 108]
 indices 2x = on/off, 2x + 1 for note ID
*/
fn get_note_encoding(note: &NoteEvent) -> Option<Vec<f32>> {
    let mut note_encoding: Vec<f32> = vec![0.0; ENCODING_LENGTH];
    let note_index: usize = (2 * (note.get_key_index() - 21) + 1) as usize;
    note_encoding[note_index] = 1.0;
    if note.is_note_on() {
        note_encoding[note_index - 1] = 1.0;
    }

    Some(note_encoding)
}

fn decode_note_encoding(encoding: &Vec<f32>, timestamp_ms: u32) -> Option<Vec<NoteEvent>> {
    if encoding.len() != ENCODING_LENGTH {
        panic!("Not a valid note encoding");
    }

    let mut events: Vec<NoteEvent> = Vec::new();

    for i in 1..encoding.len() {
        if i % 2 != 1 {
            continue;
        }
        if encoding[i] == 1.0 {
            let key: u8 = ((i - 1) / 2 + 21) as u8;
            let event: NoteEvent = NoteEvent::new(timestamp_ms, key, encoding[i - 1] == 1.0);
            events.push(event);
        }
    }

    if events.is_empty() {
        return None;
    }
    Some(events)
}

// if same note on + off, push off to next timestep? (somehow the data has 40 ms note diffs...)
fn layer_notes(n1: &Vec<f32>, n2: &Vec<f32>) -> Option<Vec<f32>> {
    if n1.len() != ENCODING_LENGTH || n2.len() != ENCODING_LENGTH {
        panic!("One (or two) invalid note encodings");
    }

    let mut new_encoding: Vec<f32> = Vec::new();

    for i in 0..ENCODING_LENGTH {
        if i % 2 != 1 {
            continue;
        }

        let mut val: f32 = n1[i] + n2[i];
        let mut on_off: f32 = n1[i - 1] + n2[i - 1];

        if val > 1.0 {
            //double note
            if on_off > 1.0 {
                // on + on
                val = 1.0;
                on_off = 1.0;
            } else if on_off == 1.0 {
                // on + off
                val = 0.0;
                on_off = 0.0;
            } else {
                // off + off
                val = 1.0;
            }
        }

        new_encoding.push(on_off);
        new_encoding.push(val);
    }

    Some(new_encoding)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi::parse_midi;

    #[test]
    fn encode_simple_midi() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");
        let data: EncodingData = EncodingData::new(&events, 500);
        let midi: MIDIEncoding = encode(data);
        println!("{}", midi.get_encoding().len());
        for (i, vector) in midi.get_encoding().iter().enumerate() {
            println!("Timestep {i}: {:?}", vector);
        }
    }

    #[test]
    fn encode_complex_midi() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Double_Note_Test.mid");
        let data: EncodingData = EncodingData::new(&events, 250);
        let midi: MIDIEncoding = encode(data);
        println!("{}", midi.get_encoding().len());
        for (i, vector) in midi.get_encoding().iter().enumerate() {
            println!("Timestep {i}: {:?}", vector);
        }
    }

    #[test]
    fn layer_test(){
        let n1: Vec<f32> = vec![1.0; 5];
        let n2: Vec<f32> = vec![2.0; 5];
        assert_eq!(layer_notes(&n1, &n2), Some(vec![3.0; 5]));
    }

    #[test]
    fn test_decode_encoding(){
        let events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");
        let data: EncodingData = EncodingData::new(&events, 500);
        let midi: MIDIEncoding = encode(data);

        let note: Option<Vec<NoteEvent>> = decode_note_encoding(&midi.get_encoding()[2], 1000);

        println!("{:?}", note);
    }

    #[test]
    fn simple_encode_and_decode(){
        let events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");
        let data: EncodingData = EncodingData::new(&events, 500);
        let encoding: MIDIEncoding = encode(data);

        let decoded: Vec<NoteEvent> = decode(encoding);
        let copy_events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");

        assert_eq!(decoded, copy_events);
    }

    #[test]
    //doesn't return same vectors bc note on & off at same timestep in original midi file (oops)
    fn complex_encode_and_decode(){
        let events: Vec<NoteEvent> = parse_midi("./tests/Double_Note_Test.mid");
        let data: EncodingData = EncodingData::new(&events, 250);
        let encoding: MIDIEncoding = encode(data);

        let mut decoded: Vec<NoteEvent> = decode(encoding);
        decoded.sort_by(|a, b| a.cmp(b));
        let mut copy_events: Vec<NoteEvent> = parse_midi("./tests/Double_Note_Test.mid");
        copy_events.sort_by(|a, b| a.cmp(b));

        println!("Decoded: {:#?}", decoded);
        println!("Original: {:#?}", copy_events);
    }

    #[test]
    fn layering_same_note() {
        let on: NoteEvent = NoteEvent::new(0, 40, true);
        let off: NoteEvent = NoteEvent::new(0, 40, false);

        let on_encoding: Vec<f32> = get_note_encoding(&on)
            .expect("Failed to get note encoding");
        let off_encoding: Vec<f32> = get_note_encoding(&off)
            .expect("Failed to get note encoding");

        let note: Vec<f32> = layer_notes(&off_encoding, &on_encoding)
            .expect("Failed to layer note encodings");

        assert_eq!(note, vec![0.0; ENCODING_LENGTH]);
    }
    
    #[test]
    fn layering_different_note() {
        let one: NoteEvent = NoteEvent::new(0, 21, true);
        let two: NoteEvent = NoteEvent::new(0, 22, true);

        let one_encoding: Vec<f32> = get_note_encoding(&one)
            .expect("Failed to get note encoding");
        let two_encoding: Vec<f32> = get_note_encoding(&two)
            .expect("Failed to get note encoding");

        let mut answer: Vec<f32> = vec![0.0; ENCODING_LENGTH];
        answer[0] = 1.0; //note on
        answer[1] = 1.0; //note 21
        answer[2] = 1.0; //note on
        answer[3] = 1.0; //note 22 

        let note: Vec<f32> = layer_notes(&one_encoding, &two_encoding)
            .expect("Failed to layer note encodings");

        assert_eq!(note , answer);
    }
}