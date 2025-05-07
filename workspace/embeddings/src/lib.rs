// builtin

// external

// internal
pub mod embedding_model;
mod types;

#[cfg(test)]
mod tests {
    use crate::{embedding_model::EmbeddingModel, types::Embedding};
    use midi_encoder::{decode_to_midi, get_sample_encoding, types::MIDIEncoding};
    use models::networks::activation::init_registry;

    #[test]
    fn embedding_training_test() {
        init_registry();

        let midi: MIDIEncoding = get_sample_encoding("../midi_encoder/tests/Data_Test.midi", 5.0);
        let mut model: EmbeddingModel = EmbeddingModel::new(32, 1, 8);

        model.learn_embeddings(&midi);

        model.save();
    }

    #[test]
    fn embedding_encode_decode_test() {
        init_registry();
        let midi: MIDIEncoding = get_sample_encoding("../midi_encoder/tests/Data_Test.midi", 5.0);
        let mut model: EmbeddingModel = EmbeddingModel::new(32, 1, 8);

        let embeddings: Embedding = model.get_embedding(&midi);

        println!("Embedding dim: {:?}", embeddings.get_dim());

        let new_midi: MIDIEncoding = model.get_encoding(embeddings);

        println!("{}", new_midi.print());

        // decode_to_midi(new_midi, "./tests/output/embed_translated.midi");
    }
}
