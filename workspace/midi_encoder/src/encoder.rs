// builtin

// external

// internal
use crate::types::{AddNoteResult, Chord, EncodingData, MIDIEncoding, NoteEvent};

pub fn encode(mut data: EncodingData) -> MIDIEncoding {
    let events: Vec<NoteEvent> = data.get_events();

    let mut time: f32 = 0.0;
    let mut encoding: Vec<Chord> = Vec::new();
    encoding.push(Chord::start());
    let chord: &mut Chord = &mut Chord::none();

    for event in events {
        let time_delta = event.get_time_delta();
        time += time_delta;
        if !data.continue_sampling(time.ceil() as u32) {
            break;
        }

        let same_timestep: bool = event.get_time_delta() == 0.0;

        if same_timestep {
            match chord.try_add(event.get_note(), time_delta) {
                AddNoteResult::Duplicate => {
                    println!("Tried to add duplicate event -- skipping");
                }
                AddNoteResult::Ok => {}
            }
        } else {
            if !chord.is_none() {
                encoding.push(chord.clone());
            }
            chord.reset();
            chord.try_add(event.get_note(), time_delta);
        }
    }

    encoding.push(chord.clone());
    encoding.push(Chord::end());

    MIDIEncoding::new(encoding)
}

pub fn decode(midi: MIDIEncoding) -> Vec<NoteEvent> {
    let mut events: Vec<NoteEvent> = Vec::new();

    for chord in midi.get_encoding().iter() {
        if chord.is_none() {
            continue;
        }

        let event: Vec<NoteEvent> = chord.get_events();

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
        let data: EncodingData = EncodingData::new(events);
        let midi: MIDIEncoding = encode(data);
        for (i, note) in midi.get_encoding().iter().enumerate() {
            println!("Chord {i}: {:?}", note);
        }
    }

    #[test]
    fn encode_complex_midi() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Double_Note_Test.mid");
        let data: EncodingData = EncodingData::new(events);
        let midi: MIDIEncoding = encode(data);
        for (i, vector) in midi.get_encoding().iter().enumerate() {
            println!("Chord {i}: {:?}", vector);
        }
    }

    #[test]
    fn simple_encode_and_decode() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");
        let data: EncodingData = EncodingData::new(events);
        let encoding: MIDIEncoding = encode(data);

        let decoded: Vec<NoteEvent> = decode(encoding);
        let copy_events: Vec<NoteEvent> = parse_midi("./tests/Timing_Test.mid");

        assert_eq!(decoded, copy_events);
    }

    #[test]
    fn complex_encode_and_decode() {
        let events: Vec<NoteEvent> = parse_midi("./tests/Double_Note_Test.mid");
        let data: EncodingData = EncodingData::new(events);
        let encoding: MIDIEncoding = encode(data);

        let decoded: Vec<NoteEvent> = decode(encoding);
        let copy_events: Vec<NoteEvent> = parse_midi("./tests/Double_Note_Test.mid");

        println!("Decoded: {:#?}", decoded);
        println!("Original: {:#?}", copy_events);
    }
}
