// builtin

// external

// internal

use std::mem::take;


#[derive(Debug)]
pub struct Embedding {
    embedding: Vec<Vec<f32>>
}

impl Embedding {
    pub fn new(data: Vec<Vec<f32>>) -> Embedding {
        Embedding {
            embedding: data
        }
    }

    pub fn get_embedding(&mut self) -> Vec<Vec<f32>> {
        take(&mut self.embedding)
    }
}