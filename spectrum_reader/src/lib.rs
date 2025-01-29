// builtin

// external

// internal
pub mod types;
mod rnn;
mod converter;

pub use crate::rnn::RNN;


#[cfg(test)]
mod tests {
    use types::{Activation, ActivationConfig, ParameterConfig, WeightConfig};

    use crate::types::activation::init_registry;

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
        let mut params: ParameterConfig = ParameterConfig::new(2, 5, 3, vec![6, 4]);
        let weights: WeightConfig = WeightConfig::new(-0.2, 0.2, -0.2, 0.2);
        let mut activations: ActivationConfig = ActivationConfig::new(Activation::none(), Activation::none());
        let mut rnn: RNN = RNN::new(&mut params, weights, &mut activations);

        let input_seq: Vec<Vec<f32>> = vec![
            vec![1.0, 1.0, 0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0, 1.0],
            vec![1.0, 1.0, 0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0, 0.0]
        ];

        let answer_seq: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0]
        ];

        for _i in 0..2000 {
            rnn.predict_and_update(input_seq.clone(), &answer_seq, 1);
        }

        let ans = rnn.predict(input_seq);

        println!("{:?}", ans);
    }
}