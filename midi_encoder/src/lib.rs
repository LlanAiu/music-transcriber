// builtin

// external

// internal
mod types;
mod constants;
mod midi;
mod encoder;
use encoder::{decode, encode};
use midi::{parse_midi, write_midi};
use types::EncodingData;

use crate::types::{NoteEvent, MIDIEncoding};


pub fn generate_midi_encoding(path: &str, timestep_ms: f32) -> MIDIEncoding {
    let events: Vec<NoteEvent> = parse_midi(path);
    let data: EncodingData = EncodingData::new(events, timestep_ms);
    encode(data)
}

pub fn decode_to_midi(midi: MIDIEncoding, file_path: &str) {
    let events: Vec<NoteEvent> = decode(midi);
    write_midi(&events, file_path);
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn simple_midi_test() {
        let midi: MIDIEncoding = generate_midi_encoding("./tests/Timing_Test.mid", 500.0);

        println!{"{:#?}", midi};
    }

    #[test]
    fn complex_midi_test() {
        let midi: MIDIEncoding = generate_midi_encoding("./tests/Double_Note_Test.mid", 250.0);

        println!("{:#?}", midi);
    }

    #[test]
    fn decode_test() {
        let midi: MIDIEncoding = generate_midi_encoding("./tests/Double_Note_Test.mid", 50.0);
        decode_to_midi(midi, "./tests/output/MGROL.mid");
    }

    #[test]
    fn data_decode_test() {
        let timestep: f32 = 512000.0 / 44100.0;
        let midi: MIDIEncoding = generate_midi_encoding("./tests/Data_Test.midi", timestep);
        decode_to_midi(midi, "./tests/output/Data_Rewrite.mid");
    }

    #[test]
    fn data_test() {
        let timestep: f32 = 512000.0 / 44100.0;
        let midi: MIDIEncoding = generate_midi_encoding("./tests/Data_Test.midi", timestep);
        println!("Number of Encodings: {}", midi.len());
    }

}
