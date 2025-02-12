// builtin

// external

// internal
mod nn;
mod types;

pub use nn::NN;

#[cfg(test)]
mod tests {

    use crate::types::{activation::init_registry, Activation, ActivationConfig, ParameterConfig, WeightConfig};

    use super::*;
    
    #[test]
    fn quadratic_test() {
        init_registry();
        let mut params: ParameterConfig = ParameterConfig::new(1, 1, 1, vec![5]);
        let weights: WeightConfig = WeightConfig::new(-0.1, 0.1, -0.001, 0.001);
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
            vec![1.0],
            vec![4.0],
            vec![9.0],
            vec![16.0],
            vec![25.0]
        ];

        for _i in 0..1000 {
            nn.predict_and_update(input_seq.clone(), &answer_seq, 5);
        }

        let test_seq: Vec<Vec<f32>> = vec![
            vec![1.5],
            vec![2.5],
            vec![3.5],
            vec![4.5],
            vec![5.5]
        ];

        let ans = nn.predict(test_seq);

        println!("{:?}", ans);
    }
}
