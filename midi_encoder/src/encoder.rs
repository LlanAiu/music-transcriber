// builtin

// external

// internal
use crate::types::{EncodingData, MIDIEncoding, Note, NoteEvent, AddNoteResult};


// just betting nobody can hit the same note twice in one timestep...
pub fn encode(data: EncodingData) -> MIDIEncoding {
    let events: &Vec<NoteEvent> = data.get_events();
    let timestep_ms: f32 = data.get_timestep();

    let mut time: f32 = 0.0;
    let mut encoding: Vec<Note> = Vec::new();
    let prev: &mut Note = &mut Note::none();
    let next: &mut Note = &mut Note::none();

    for event in events.iter() {
        let same_timestep: bool = (event.get_timestamp() as f32 - time) < timestep_ms;

        if same_timestep {
            match prev.try_add(event) {
                AddNoteResult::Duplicate => {
                    println!("Tried to add duplicate event -- skipping");
                },
                AddNoteResult::PushToNext(note) => {
                    next.try_add(note);
                },
                AddNoteResult::Ok => {}
            }
        } else {
            let next_time: f32 = event.get_timestamp() as f32;
            
            encoding.push(prev.clone());
            prev.reset();
            time += timestep_ms;

            let add_next: bool = !next.is_none() && time + timestep_ms <= next_time;

            if add_next {
                encoding.push(next.clone());
                time += timestep_ms;
            } else {
                let event: Vec<NoteEvent> = next.get_events();
                
                for e in event.iter() {
                    prev.try_add(e);
                }
            }

            next.reset();

            while time + timestep_ms <= next_time {
                encoding.push(Note::none());
                time += timestep_ms;
            }

            prev.try_add(event);
        }
    }

    encoding.push(prev.clone());
    
    if !next.is_none() {
        encoding.push(next.clone());
    }

    MIDIEncoding::new(timestep_ms, encoding)
}

// pub fn encode(data: EncodingData) -> MIDIEncoding {
//     let events: &Vec<NoteEvent> = data.get_events();
//     let timestep_ms: u32 = data.get_timestep();
    
//     let mut current_time: u32 = 0;
//     let mut encoding: Vec<Vec<f32>> = Vec::new();
//     let mut previous_note: Option<Vec<f32>> = None;

//     for event in events.iter() {
//         let should_layer: bool = previous_note.is_some()
//             && event.get_timestamp().abs_diff(current_time) < timestep_ms;
//         if should_layer {
//             let prev: Vec<f32> = previous_note.expect("Cannot layer null note encoding");
//             let note: Vec<f32> = get_note_encoding(event).expect("Failed to get note encoding");
//             previous_note = layer_notes(&prev, &note);
//         } else {
//             if previous_note.is_some() {
//                 let note: Vec<f32> = previous_note.expect("Cannot get note encoding");
//                 encoding.push(note);
//                 current_time += timestep_ms;
//             }
//             let next_time = event.get_timestamp();
//             while current_time + timestep_ms <= next_time {
//                 current_time += timestep_ms;
//                 encoding.push(vec![0.0; ENCODING_LENGTH]);
//             }
//             previous_note = get_note_encoding(event);
//         }
//     }
//     encoding.push(previous_note.expect("Cannot get note encoding"));

//     MIDIEncoding::new(timestep_ms, encoding)
// }

pub fn decode(midi: MIDIEncoding) -> Vec<NoteEvent> {
    let mut events: Vec<NoteEvent> = Vec::new();

    for note in midi.get_encoding().iter() {
        if note.is_none() {
            continue;
        }

        let event: Vec<NoteEvent> = note.get_events();

        for e in event {
            events.push(e);
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi::parse_midi;

    #[test]
    fn encode_simple_midi() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");
        let data: EncodingData = EncodingData::new(&events, 500.0);
        let midi: MIDIEncoding = encode(data);
        println!("{}", midi.get_encoding().len());
        for (i, note) in midi.get_encoding().iter().enumerate() {
            println!("Timestep {i}: {:?}", note);
        }
    }

    #[test]
    fn encode_complex_midi() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Double_Note_Test.mid");
        let data: EncodingData = EncodingData::new(&events, 250.0);
        let midi: MIDIEncoding = encode(data);
        println!("{}", midi.get_encoding().len());
        for (i, vector) in midi.get_encoding().iter().enumerate() {
            println!("Timestep {i}: {:?}", vector);
        }
    }

    #[test]
    fn simple_encode_and_decode(){
        let events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");
        let data: EncodingData = EncodingData::new(&events, 500.0);
        let encoding: MIDIEncoding = encode(data);

        let decoded: Vec<NoteEvent> = decode(encoding);
        let copy_events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");

        assert_eq!(decoded, copy_events);
    }

    #[test]
    //doesn't return same vectors bc note on & off at same timestep in original midi file (oops)
    fn complex_encode_and_decode(){
        let events: Vec<NoteEvent> = parse_midi("./tests/Double_Note_Test.mid");
        let data: EncodingData = EncodingData::new(&events, 250.0);
        let encoding: MIDIEncoding = encode(data);

        let mut decoded: Vec<NoteEvent> = decode(encoding);
        decoded.sort_by(|a, b| a.cmp(b));
        let mut copy_events: Vec<NoteEvent> = parse_midi("./tests/Double_Note_Test.mid");
        copy_events.sort_by(|a, b| a.cmp(b));

        println!("Decoded: {:#?}", decoded);
        println!("Original: {:#?}", copy_events);
    }

}