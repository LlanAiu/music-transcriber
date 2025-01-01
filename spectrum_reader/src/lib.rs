// builtin

// external

// internal
mod types;
mod simple_rnn;
use crate::simple_rnn::RNN;


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    pub fn construct_test(){
        let rnn: RNN = RNN::new(1, 10, 2, vec![5]);
        rnn.save_to_file("./tests/weights.txt");
    }

    #[test]
    pub fn save_load_test(){
        let rnn: RNN = RNN::from_save("./tests/weights.txt");
        rnn.save_to_file("./tests/identical_weights.txt");
    }
}