// builtin

// external

// internal
mod types;
mod simple_rnn;
use crate::simple_rnn::RNN;


#[cfg(test)]
mod tests {
    use types::ParameterConfig;

    use super::*;


    #[test]
    pub fn construct_test() {
        let params: ParameterConfig = ParameterConfig::new(-0.3, 0.3, -0.3, 0.3);
        let rnn: RNN = RNN::new(1, 10, 2, vec![5], params);
        rnn.save_to_file("./tests/weights.txt");
    }

    #[test]
    pub fn save_load_test() {
        let rnn: RNN = RNN::from_save("./tests/weights.txt");
        rnn.save_to_file("./tests/identical_weights.txt");
    }

    #[test]
    pub fn predict_test() {
        let params: ParameterConfig = ParameterConfig::new(0.999, 1.0, -0.01, 0.01);
        let rnn: RNN = RNN::new(1, 10, 2, vec![5], params);
        let input_seq: Vec<Vec<f32>> = vec![
            vec![1.0; 10],
            vec![0.0; 10],
            vec![1.0; 10]
        ];

        let output_seq: Vec<Vec<f32>> = rnn.predict(input_seq);

        println!("{:#?}", output_seq);
    }
}