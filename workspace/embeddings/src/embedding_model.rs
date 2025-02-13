// builtin
use std::fs;

// external

// internal
use midi_encoder::types::MIDIEncoding;
use models::networks::activation::Activation;
use models::NN;
use models::networks::configs::{ActivationConfig, ParameterConfig, WeightConfig};
use super::types::Embedding;


pub struct EmbeddingModel {
    dim: usize,
    window: usize,
    batch: usize,
    nn: NN,
}

impl EmbeddingModel {
    pub fn new(dim: usize, window: usize, batch: usize) -> EmbeddingModel {
        let file_path: String = format!("./tests/weights_d{dim}.txt");
        let nn: NN = NN::from_save(&file_path).unwrap_or_else(| _err | {
            let mut params: ParameterConfig = ParameterConfig::new(1, 176, 176, vec![dim]);
            let weights: WeightConfig = WeightConfig::new(0.03, 0.07, 0.000, 0.0001);
            let mut activations: ActivationConfig = ActivationConfig::new(Activation::relu(), Activation::sigmoid());
            NN::new(&mut params, weights, &mut activations)
        });

        EmbeddingModel {
            dim,
            window,
            batch,
            nn
        }
    }

    pub fn save(&self) {
        let file_path: String = format!("./tests/weights_d{}.txt", self.dim);

        self.nn.save_to_file(&file_path);
    }

    fn learn_embeddings(&self, encoding: MIDIEncoding) {
        let encoded_vecs: Vec<Vec<f32>> = encoding.get_encoding().iter().map(
            |c| c.get_encoding()
        ).collect();

        todo!()
    }
    
    pub fn get_embedding(encoding: MIDIEncoding) -> Embedding {
        todo!()
    }
    
    pub fn get_encoding(embedding: Embedding) -> MIDIEncoding {
        todo!()
    }
}
