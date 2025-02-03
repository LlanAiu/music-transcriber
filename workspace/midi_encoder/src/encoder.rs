// builtin

// external

// internal
use crate::types::{AddNoteResult, Chord, EncodingData, MIDIEncoding, Note, NoteEvent};


// just betting nobody can hit the same note twice in one timestep...
pub fn encode(data: EncodingData) -> MIDIEncoding {
    let timestep_ms: f32 = data.get_timestep();
    let events: Vec<NoteEvent> = data.get_events();

    let mut time: f32 = 0.0;
    let mut encoding: Vec<Chord> = Vec::new();
    let prev: &mut Chord = &mut Chord::none();
    let next: &mut Chord = &mut Chord::none();

    for event in events {
        let same_timestep: bool = (event.get_timestamp() as f32 - time) < timestep_ms;

        if same_timestep {
            match prev.try_add(event.get_note()) {
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
                let event: Vec<Note> = next.get_notes();
                
                for e in event {
                    prev.try_add(e);
                }
            }

            next.reset();

            while time + timestep_ms <= next_time {
                encoding.push(Chord::none());
                time += timestep_ms;
            }

            prev.try_add(event.get_note());
        }
    }

    encoding.push(prev.clone());
    
    if !next.is_none() {
        encoding.push(next.clone());
    }

    MIDIEncoding::new(timestep_ms, encoding)
}

pub fn decode(midi: MIDIEncoding) -> Vec<NoteEvent> {
    let mut events: Vec<NoteEvent> = Vec::new();

    for (i, chord) in midi.get_encoding().iter().enumerate() {
        if chord.is_none() {
            continue;
        }

        let timestamp_ms: u32 = ((i as f32) * midi.get_timestep()) as u32;
        let event: Vec<NoteEvent> = chord.get_events(timestamp_ms);

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
        let data: EncodingData = EncodingData::new(events, 500.0);
        let midi: MIDIEncoding = encode(data);
        println!("{}", midi.get_encoding().len());
        for (i, note) in midi.get_encoding().iter().enumerate() {
            println!("Timestep {i}: {:?}", note);
        }
    }

    #[test]
    fn encode_complex_midi() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Double_Note_Test.mid");
        let data: EncodingData = EncodingData::new(events, 250.0);
        let midi: MIDIEncoding = encode(data);
        println!("{}", midi.get_encoding().len());
        for (i, vector) in midi.get_encoding().iter().enumerate() {
            println!("Timestep {i}: {:?}", vector);
        }
    }

    #[test]
    fn simple_encode_and_decode(){
        let events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");
        let data: EncodingData = EncodingData::new(events, 500.0);
        let encoding: MIDIEncoding = encode(data);

        let decoded: Vec<NoteEvent> = decode(encoding);
        let copy_events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");

        assert_eq!(decoded, copy_events);
    }

    #[test]
    fn complex_encode_and_decode(){
        let events: Vec<NoteEvent> = parse_midi("./tests/Double_Note_Test.mid");
        let data: EncodingData = EncodingData::new(events, 250.0);
        let encoding: MIDIEncoding = encode(data);

        let mut decoded: Vec<NoteEvent> = decode(encoding);
        decoded.sort_by(|a, b| a.cmp(b));
        let mut copy_events: Vec<NoteEvent> = parse_midi("./tests/Double_Note_Test.mid");
        copy_events.sort_by(|a, b| a.cmp(b));

        println!("Decoded: {:#?}", decoded);
        println!("Original: {:#?}", copy_events);

        assert_eq!(decoded, copy_events);
    }

}