// builtin

// external

// internal
mod types;
mod constants;
mod midi;
mod encoder;
use encoder::{decode, encode};
use midi::parse_midi;
use types::EncodingData;

use crate::types::{NoteEvent, MIDIEncoding};


pub fn generate_midi_encoding(path: &str, timestep_ms: u32) -> MIDIEncoding {
    let events: Vec<NoteEvent> = parse_midi(path);
    let data: EncodingData = EncodingData::new(&events, timestep_ms);
    encode(data)
}

//TODO: generate midi file from encoding (change test too)
pub fn decode_midi_encoding(midi: MIDIEncoding) -> Vec<NoteEvent> {
    decode(midi)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn simple_midi_test() {
        let midi: MIDIEncoding = generate_midi_encoding("./tests/Timing_Test.mid", 500);

        println!{"{:#?}", midi};
    }

    #[test]
    fn complex_midi_test() {
        let midi: MIDIEncoding = generate_midi_encoding("./tests/Double_Note_Test.mid", 250);

        println!("{:#?}", midi);
    }

    #[test]
    fn decode_test() {
        let midi: MIDIEncoding = generate_midi_encoding("./tests/Double_Note_Test.mid", 250);
        let events: Vec<NoteEvent> = decode_midi_encoding(midi);

        println!("{:#?}", events);
    }

}
