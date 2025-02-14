// builtin
use std::cmp;
use std::os::windows;

// external

// internal
use super::types::Embedding;
use midi_encoder::types::MIDIEncoding;
use models::networks::activation::Activation;
use models::networks::configs::{ActivationConfig, ParameterConfig, WeightConfig};
use models::NN;
use ndarray::Array1;

pub struct EmbeddingModel {
    dim: usize,
    window: usize,
    batch: usize,
    nn: NN,
}

impl EmbeddingModel {
    pub fn new(dim: usize, window: usize, batch: usize) -> EmbeddingModel {
        let file_path: String = format!("./tests/weights_d{dim}.txt");
        let nn: NN = NN::from_save(&file_path).unwrap_or_else(|_err| {
            let mut params: ParameterConfig = ParameterConfig::new(1, 176, 176, vec![dim]);
            let weights: WeightConfig = WeightConfig::new(0.03, 0.07, 0.000, 0.0001);
            let mut activations: ActivationConfig =
                ActivationConfig::new(Activation::relu(), Activation::sigmoid());
            NN::new(&mut params, weights, &mut activations)
        });

        EmbeddingModel {
            dim,
            window,
            batch,
            nn,
        }
    }

    pub fn save(&self) {
        let file_path: String = format!("./tests/weights_d{}.txt", self.dim);

        self.nn.save_to_file(&file_path);
    }

    fn learn_embeddings(&mut self, encoding: MIDIEncoding) {
        let encoded_vecs: Vec<Vec<f32>> = encoding
            .get_encoding()
            .iter()
            .map(|c| c.get_encoding())
            .collect();

        let mut averaged_vecs: Vec<Vec<f32>> = Vec::new();

        for i in 0..encoded_vecs.len() {
            let start = if i >= self.window { i - self.window } else { 0 };
            let end = cmp::min(i + self.window, encoded_vecs.len() - 1);
            let window = &encoded_vecs[start..=end];

            let sum_vec = self.get_window_average(window);

            averaged_vecs.push(sum_vec);
        }

        self.nn.predict_and_update(averaged_vecs, &encoded_vecs, self.batch);
    }

    pub fn get_embedding(encoding: MIDIEncoding) -> Embedding {
        todo!()
    }

    pub fn get_encoding(embedding: Embedding) -> MIDIEncoding {
        todo!()
    }

    fn get_window_average(&self, slice: &[Vec<f32>]) -> Vec<f32> {
        if slice.len() != (2 * self.window + 1) || slice.len() % 2 == 0 {
            panic!("Invalid encoding window!");
        }

        let mut sum_vec: Vec<f32> = vec![0.0; slice[0].len()];
        let skip_i: usize = slice.len() / 2;
        let num_vecs: usize = slice.len() - 1;

        for (i, vec) in slice.iter().enumerate() {
            if i == skip_i {
                continue;
            }

            for j in 0..sum_vec.len() {
                sum_vec[j] += vec[j] / num_vecs as f32;
            }
        }

        sum_vec
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn window_average_test() {
        let model = EmbeddingModel::new(10, 3, 1);
        let input: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![7.0, 8.0, 9.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
        ];

        let slice = &input[0..7];
        let expected_output = vec![1.0, 2.0, 3.0];
        let result = model.get_window_average(slice);
        assert_eq!(result, expected_output);
    }
}
