// builtin

// external

// internal
pub mod types;
mod rnn;
mod converter;

pub use crate::converter::RNNConverter;
pub use crate::rnn::RNN;


#[cfg(test)]
mod tests {
    use audio_to_spectrum::{get_sample_spectrograph, spectrograph::Spectrograph};
    use midi_encoder::{get_sample_encoding, types::MIDIEncoding};
    use types::{Activation, ActivationConfig, ParameterConfig, WeightConfig};

    use crate::{converter::Translator, types::{activation::init_registry, ConverterConfig}};

    use super::*;

    #[test]
    pub fn construct_test() {
        init_registry();
        let mut params: ParameterConfig = ParameterConfig::new(1, 10, 2, vec![5]);
        let weights: WeightConfig = WeightConfig::new(-0.3, 0.3, -0.3, 0.3);
        let mut activations: ActivationConfig = ActivationConfig::new(Activation::relu(), Activation::sigmoid());
        let rnn: RNN = RNN::new(&mut params, weights, &mut activations);
        rnn.save_to_file("./tests/weights.txt");
    }

    #[test]
    pub fn save_load_test() {
        init_registry();
        init_registry();
        let rnn: RNN = RNN::from_save("./tests/weights.txt");
        rnn.save_to_file("./tests/identical_weights.txt");
    }

    #[test]
    pub fn predict_test() {
        init_registry();
        let mut params: ParameterConfig = ParameterConfig::new(1, 10, 2, vec![5]);
        let weights: WeightConfig = WeightConfig::new(0.999, 1.0, -0.01, 0.01);
        let mut activations: ActivationConfig = ActivationConfig::new(Activation::none(), Activation::none());
        let mut rnn: RNN = RNN::new(&mut params, weights, &mut activations);
        let input_seq: Vec<Vec<f32>> = vec![
            vec![1.0; 10],
            vec![0.0; 10],
            vec![1.0; 10]
        ];

        let output_seq: Vec<Vec<f32>> = rnn.predict(input_seq);

        println!("{:#?}", output_seq);
    }

    #[test]
    pub fn convergence_test() {
        init_registry();
        let mut params: ParameterConfig = ParameterConfig::new(2, 5, 3, vec![4, 2]);
        let weights: WeightConfig = WeightConfig::new(-0.05, 0.05, -0.05, 0.05);
        let mut activations: ActivationConfig = ActivationConfig::new(Activation::none(), Activation::none());
        let mut rnn: RNN = RNN::new(&mut params, weights, &mut activations);

        let input_seq: Vec<Vec<f32>> = vec![
            vec![1.0, 1.0, 0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0, 1.0]
        ];

        let answer_seq: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0]
        ];

        for _i in 0..1000 {
            rnn.predict_and_update(input_seq.clone(), &answer_seq, 3);
        }

        let ans = rnn.predict(input_seq);

        println!("{:?}", ans);
    }

    #[test]
    fn quadratic_interpolation_test() {
        let mut params: ParameterConfig = ParameterConfig::new(1, 1, 1, vec![4]);
        let weights: WeightConfig = WeightConfig::new(0.01, 0.05, 0.000, 0.001);
        let mut activations: ActivationConfig = ActivationConfig::new(Activation::relu(), Activation::none());
        let mut rnn: RNN = RNN::new(&mut params, weights, &mut activations);

        let input_seq: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0]
        ];

        let answer_seq: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![4.0],
            vec![9.0],
            vec![16.0],
            vec![25.0]
        ];

        for _i in 0..1000 {
            rnn.predict_and_update(input_seq.clone(), &answer_seq, 5);
        }

        let test_seq: Vec<Vec<f32>> = vec![
            vec![1.5],
            vec![2.5],
            vec![3.5],
            vec![4.5],
            vec![5.5]
        ];

        let ans = rnn.predict(test_seq);

        println!("{:?}", ans);
    }

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