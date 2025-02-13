// builtin

// external

// internal


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
}