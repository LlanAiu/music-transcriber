// builtin

// external

// internal

use std::mem::take;

#[derive(Debug)]
pub struct Embedding {
    embedding: Vec<Vec<f32>>,
}

impl Embedding {
    pub fn new(data: Vec<Vec<f32>>) -> Embedding {
        Embedding { embedding: data }
    }

    pub fn get_dim(&self) -> (usize, usize) {
        let embedding_num: usize = self.embedding.len();
        let null_vec: Vec<f32> = vec![];
        let first_embedding: &Vec<f32> = self.embedding.get(0).unwrap_or(&null_vec);
        let embedding_dim: usize = first_embedding.len();

        return (embedding_num, embedding_dim);
    }

    pub fn get_embedding(&mut self) -> Vec<Vec<f32>> {
        take(&mut self.embedding)
    }
}
