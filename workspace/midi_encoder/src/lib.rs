// builtin

// external

// internal
mod constants;
mod encoder;
mod midi;
pub mod types;
use encoder::{decode, encode};
use midi::{parse_midi, write_midi};
use types::{EncodingData, MIDIEncoding, NoteEvent};

pub fn generate_midi_encoding(path: &str) -> MIDIEncoding {
    let events: Vec<NoteEvent> = parse_midi(path);
    let data: EncodingData = EncodingData::new(events);
    encode(data)
}

pub fn get_sample_encoding(path: &str, len_sec: f32) -> MIDIEncoding {
    let limit_ms: usize = (len_sec * 1000.0).floor() as usize;
    let events: Vec<NoteEvent> = parse_midi(path);
    let data: EncodingData = EncodingData::with_limit(events, limit_ms);
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
        let midi: MIDIEncoding = generate_midi_encoding("./tests/Timing_Test.mid");

        println! {"{:?}", midi};
    }

    #[test]
    fn complex_midi_test() {
        let midi: MIDIEncoding = generate_midi_encoding("./tests/Double_Note_Test.mid");

        println!("{:?}", midi);
    }

    #[test]
    fn decode_test() {
        let midi: MIDIEncoding = generate_midi_encoding("./tests/Double_Note_Test.mid");
        decode_to_midi(midi, "./tests/output/MGROL.mid");
    }

    #[test]
    fn data_decode_test() {
        let midi: MIDIEncoding = generate_midi_encoding("./tests/Data_Test.midi");
        decode_to_midi(midi, "./tests/output/Data_Rewrite.mid");
    }

    #[test]
    fn data_test() {
        let midi: MIDIEncoding = generate_midi_encoding("./tests/Data_Test.midi");
        println!("Number of Encodings: {}", midi.len());
    }

    #[test]
    fn sample_data_test() {
        let midi: MIDIEncoding = get_sample_encoding("./tests/Data_Test.midi", 5.0);
        println!("Number of Encodings: {}", midi.len());
        println!("{}", midi.print());
        decode_to_midi(midi, "./tests/output/Data_truncated.mid");
    }
}
