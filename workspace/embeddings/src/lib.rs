// builtin

// external

// internal
pub mod embedding_model;
mod types;

#[cfg(test)]
mod tests {
    use crate::embedding_model::EmbeddingModel;
    use midi_encoder::{get_sample_encoding, types::MIDIEncoding};

    #[test]
    fn embedding_model_test() {
        let midi: MIDIEncoding = get_sample_encoding("../midi_encoder/tests/Data_Test.midi", 5.0);
        let mut model: EmbeddingModel = EmbeddingModel::new(32, 1, 8);

        model.learn_embeddings(midi);

        model.save();
    }
}
