// builtin

// external

// internal
mod types;
mod simple_rnn;

pub use crate::simple_rnn::RNN;


#[cfg(test)]
mod tests {
    use types::{Activation, ActivationConfig, ParameterConfig, WeightConfig};

    use super::*;

    #[test]
    pub fn construct_test() {
        let mut params: ParameterConfig = ParameterConfig::new(1, 10, 2, vec![5]);
        let weights: WeightConfig = WeightConfig::new(-0.3, 0.3, -0.3, 0.3);
        let mut activations: ActivationConfig = ActivationConfig::new(Activation::relu(), Activation::sigmoid());
        let rnn: RNN = RNN::new(&mut params, weights, &mut activations);
        rnn.save_to_file("./tests/weights.txt");
    }

    #[test]
    pub fn save_load_test() {
        let rnn: RNN = RNN::from_save("./tests/weights.txt");
        rnn.save_to_file("./tests/identical_weights.txt");
    }

    #[test]
    pub fn predict_test() {
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
}