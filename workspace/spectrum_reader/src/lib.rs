// builtin

// external

// internal
pub mod types;
mod converter;

pub use crate::converter::RNNConverter;


#[cfg(test)]
mod tests {
    use audio_to_spectrum::{get_sample_spectrograph, spectrograph::Spectrograph};
    use midi_encoder::{get_sample_encoding, types::MIDIEncoding};
    use models::networks::configs::{ActivationConfig, WeightConfig};
    use models::networks::activation::{Activation, init_registry};

    use crate::types::{Translator, ConverterConfig};

    use super::*;

    #[test]
    fn new_data_test() {
        let config: ConverterConfig = ConverterConfig::new(1, vec![50], 6);
        let weights: WeightConfig = WeightConfig::new(0.05, 0.2, -0.4, 0.4);
        let activations: ActivationConfig = ActivationConfig::new(Activation::relu(), Activation::sigmoid());
        
        let mut converter: RNNConverter = RNNConverter::new(config, weights, activations);
        
        let graph: Spectrograph = get_sample_spectrograph("./tests/Data_test.mp3", 3.0);
        let encoding: MIDIEncoding = get_sample_encoding("./tests/Data_test.midi", 2.99);

        for _i in 1..1 {
            converter.update(graph.clone(), encoding.clone());
        }

        let output: MIDIEncoding = converter.translate_spectrum(graph, 0.7);

        converter.save("./tests/converter_weights.txt");

        println!("{}", output.print());
    }

    #[test]
    fn save_data_test() {
        init_registry();
        let mut converter: RNNConverter = RNNConverter::from_file("./tests/converter_weights.txt", 6);
        
        let graph: Spectrograph = get_sample_spectrograph("./tests/Data_test.mp3", 3.0);
        let encoding: MIDIEncoding = get_sample_encoding("./tests/Data_test.midi", 2.99);

        for _i in 1..2 {
            converter.update(graph.clone(), encoding.clone());
        }

        let output: MIDIEncoding = converter.translate_spectrum(graph, 0.7);

        converter.save("./tests/converter_weights.txt");

        println!("{}", output.print());
    }
}