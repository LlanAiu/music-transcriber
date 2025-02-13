// builtin

// external

// internal
mod nn;
mod rnn;
pub mod networks;

pub use nn::NN;
pub use rnn::RNN;

#[cfg(test)]
mod tests {

    use crate::networks::{activation::{init_registry, Activation}, configs::{ActivationConfig, ParameterConfig, WeightConfig}};

    use super::*;

    #[test]
    fn linear_test() {
        init_registry();
        let mut params: ParameterConfig = ParameterConfig::new(1, 1, 1, vec![1]);
        let weights: WeightConfig = WeightConfig::new(0.03, 0.07, 0.000, 0.001);
        let mut activations: ActivationConfig = ActivationConfig::new(Activation::relu(), Activation::none());
        let mut nn: NN = NN::new(&mut params, weights, &mut activations);

        let input_seq: Vec<Vec<f32>> = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0]
        ];

        let answer_seq: Vec<Vec<f32>> = vec![
            vec![2.0],
            vec![4.0],
            vec![6.0],
            vec![8.0],
            vec![10.0]
        ];

        for _i in 0..1000 {
            nn.predict_and_update(input_seq.clone(), &answer_seq, 5);
        }

        let test_seq: Vec<Vec<f32>> = vec![
            vec![1.5],
            vec![2.0],
            vec![2.5],
            vec![3.0],
            vec![3.5]
        ];

        let ans = nn.predict(test_seq);

        println!("{:?}", ans);
    }
    
    #[test]
    fn quadratic_test() {
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
}
