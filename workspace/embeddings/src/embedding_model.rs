// builtin
use std::cmp;

// external

// internal
use super::types::Embedding;
use midi_encoder::types::{MIDIEncoding, ENCODING_LENGTH};
use models::networks::activation::Activation;
use models::networks::configs::{ActivationConfig, ParameterConfig, WeightConfig};
use models::NN;

pub struct EmbeddingModel {
    dim: usize,
    window_radius: usize,
    batch: usize,
    nn: NN,
}

impl EmbeddingModel {
    pub fn new(dim: usize, window: usize, batch: usize) -> EmbeddingModel {
        let file_path: String = format!("./tests/weights_d{dim}.txt");
        let nn: NN = NN::from_save(&file_path).unwrap_or_else(|_err| {
            let mut params: ParameterConfig =
                ParameterConfig::new(1, ENCODING_LENGTH, ENCODING_LENGTH, vec![dim]);
            let weights: WeightConfig = WeightConfig::new(0.03, 0.07, 0.000, 0.0001);
            let mut activations: ActivationConfig =
                ActivationConfig::new(Activation::relu(), Activation::sigmoid());
            NN::new(&mut params, weights, &mut activations)
        });

        EmbeddingModel {
            dim,
            window_radius: window,
            batch,
            nn,
        }
    }

    pub fn save(&self) {
        let file_path: String = format!("./tests/weights_d{}.txt", self.dim);

        self.nn.save_to_file(&file_path);
    }

    pub fn learn_embeddings(&mut self, encoding: MIDIEncoding) {
        let encoded_vecs: Vec<Vec<f32>> = encoding
            .get_encoding()
            .iter()
            .map(|c| c.get_encoding())
            .collect();

        let mut averaged_vecs: Vec<Vec<f32>> = Vec::new();

        for i in 0..encoded_vecs.len() {
            let start = if i >= self.window_radius {
                i - self.window_radius
            } else {
                0
            };
            let end = cmp::min(i + self.window_radius, encoded_vecs.len() - 1);
            let window = &encoded_vecs[start..=end];

            let sum_vec = self.get_window_average(window);

            averaged_vecs.push(sum_vec);
        }

        self.nn
            .predict_and_update(averaged_vecs, &encoded_vecs, self.batch);
    }

    pub fn get_embedding(&mut self, encoding: MIDIEncoding) -> Embedding {
        let encoded_vecs: Vec<Vec<f32>> = encoding
            .get_encoding()
            .iter()
            .map(|c| c.get_encoding())
            .collect();

        let output_vecs: Vec<Vec<f32>> = self.nn.predict_first_layer(encoded_vecs);

        Embedding::new(output_vecs)
    }

    pub fn get_encoding(&mut self, mut embedding: Embedding) -> MIDIEncoding {
        let embedding_vecs: Vec<Vec<f32>> = embedding.get_embedding();

        let output_vecs: Vec<Vec<f32>> = self.nn.first_layer_input(embedding_vecs);

        MIDIEncoding::from_vector(output_vecs, 0.5)
    }

    fn get_window_average(&self, slice: &[Vec<f32>]) -> Vec<f32> {
        if slice.is_empty() {
            panic!("Empty window!");
        }

        let mut sum_vec: Vec<f32> = vec![0.0; slice[0].len()];
        let center_index: usize = slice.len() / 2;
        let num_vecs: usize = 2 * self.window_radius;

        for (i, vec) in slice.iter().enumerate() {
            if i == center_index {
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
