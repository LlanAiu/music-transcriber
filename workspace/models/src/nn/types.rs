// builtin 

// external
use ndarray::{Array, Array1, Array2, ArrayView2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

// internal


pub struct Bias {
    array: Array1<f32>
}

impl Bias {
    pub fn new(array: Array1<f32>) -> Bias {
        Bias { array }
    }

    pub fn random(size: usize, min: f32, max: f32) -> Bias {
        let array: Array1<f32> = Array::random(size, Uniform::new(min, max));
        Bias::new(array)
    }

    pub fn from_string(s: &str) -> Bias {
        let split: Vec<&str> = s.split("#").collect();

        let dim: usize = split[0].parse().expect("Failed to parse array dimension");

        let vec: Vec<f32> = split[1].split(",").map(|num| {
            num.parse::<f32>().expect("Failed to parse string as f32")
        }).collect();

        if dim != vec.len() {
            panic!("Length mismatch of bias data");
        }

        Bias::from_vec(vec)
    }

    pub fn from_vec(vec: Vec<f32>) -> Bias {
        let array: Array1<f32> = Array1::from_vec(vec);
        Bias { array }
    }

    pub fn get_row_vector(&self) -> ArrayView2<f32> {
        self.array.view().insert_axis(Axis(0))
    }
    
    // pub fn get_column_vector(&self) -> ArrayView2<f32> {
    //     self.array.view().insert_axis(Axis(1))
    // }

    pub fn update(&mut self, update: &Array1<f32>, scale: f32) {
        let scaled: Array1<f32> = scale * update;
        self.array -= &scaled;
    }
}

impl ToString for Bias {
    fn to_string(&self) -> String {
        let dim = self.array.dim();

        let mut formatted: String = format!("{}#", dim);

        let val_string: &str = &self.array.iter()
            .map(|w| w.to_string())
            .collect::<Vec<String>>()
            .join(",");

        formatted.push_str(val_string);

        formatted
    }
}

pub struct Weight {
    array: Array2<f32>
}

impl Weight {
    pub fn new(array: Array2<f32>) -> Weight {
        Weight { array }
    }

    pub fn random(dim: (usize, usize), min: f32, max: f32) -> Weight {
        let array: Array2<f32> = Array::random(dim, Uniform::new(min, max));
        Weight::new(array)
    }

    pub fn from_string(s: &str) -> Weight {
        let split: Vec<&str> = s.split("#").collect();

        let dims: Vec<usize> = split[0].split(",").map(|num| {
            num.parse::<usize>().expect("Failed to parse string as usize")
        }).collect();

        let vec: Vec<f32> = split[1].split(",").map(|num| {
            num.parse::<f32>().expect("Failed to parse string as f32")
        }).collect();

        if dims[0] * dims[1] != vec.len() {
            panic!("Length mismatch of weight data");
        }

        Weight::from_vec((dims[0], dims[1]), vec)
    }

    pub fn from_vec(shape: (usize, usize), vec: Vec<f32>) -> Weight {
        let array: Array2<f32> = Array2::from_shape_vec(shape, vec).expect("Failed to construct array");
        Weight { array }
    }

    pub fn get_weight_matrix(&self) -> ArrayView2<f32> {
        self.array.view()
    }

    pub fn update(&mut self, update: &Array2<f32>, alpha: f32) {
        let scaled: Array2<f32> = alpha * update;
        self.array -= &scaled;
    }

    pub fn dim(&self) -> (usize, usize) {
        self.array.dim()
    }
}

impl ToString for Weight {
    fn to_string(&self) -> String {
        let (dim1, dim2) = self.array.dim();

        let mut formatted: String = format!("{},{}#", dim1, dim2);

        let val_string: &str = &self.array.iter()
            .map(|w| w.to_string())
            .collect::<Vec<String>>()
            .join(",");

        formatted.push_str(val_string);

        formatted
    }
}

#[derive(Debug)]
pub struct Update {
    batch_count: usize,
    max_batch_size: usize,
    hidden_update: Vec<Array2<f32>>,
    biases_update: Vec<Array1<f32>>
}

impl Update {
    pub fn new(
        input_size: usize, 
        output_size: usize, 
        units_by_layer: &Vec<usize>, 
        max_batch_size: usize
    ) -> Update {
        let layers: usize = units_by_layer.len();
        let mut hidden_update: Vec<Array2<f32>> = Vec::with_capacity(layers + 1);
        let mut biases_update: Vec<Array1<f32>> = Vec::with_capacity(layers + 1);

        for i in 0..=layers {
            let dim1: usize;
            let dim2: usize;

            if i == 0 {
                dim1 = input_size;
                dim2 = *units_by_layer.get(i).expect("Failed to get layer dimension");
            } else if i == layers {
                dim1 = *units_by_layer.get(i - 1).expect("Failed to get layer dimension");
                dim2 = output_size;
            } else {
                dim1 = *units_by_layer.get(i - 1).expect("Failed to get layer dimension");
                dim2 = *units_by_layer.get(i).expect("Failed to get layer dimension");
            }

            let hidden: Array2<f32> = Array2::zeros((dim1, dim2));
            hidden_update.push(hidden);

            let bias: Array1<f32> = Array1::zeros(dim2);
            biases_update.push(bias);
        }

        Update {
            batch_count: 0,
            max_batch_size,
            hidden_update,
            biases_update
        }
    }

    pub fn should_update(&self) -> bool {
        self.batch_count >= self.max_batch_size
    }

    pub fn combine_update(&mut self, hidden: Vec<Array2<f32>>, biases: Vec<Array1<f32>>) {
        self.batch_count += 1;

        if hidden.len() != self.hidden_update.len() || 
            biases.len() != self.biases_update.len() 
        {
            panic!("Dimension mismatch between existing and new updates");
        }

        for (i, update) in hidden.iter().enumerate() {
            self.hidden_update[i] += update;
        }

        for (i, update) in biases.iter().enumerate() {
            self.biases_update[i] += update;
        }
    }

    pub fn get_hidden_update(&self) -> &Vec<Array2<f32>> {
        &self.hidden_update
    }

    pub fn get_biases_update(&self) -> &Vec<Array1<f32>> {
        &self.biases_update
    }

    pub fn get_norm(&self) -> f32 {
        let hidden_norm: f32 = self.hidden_update.iter()
            .map(|arr| arr.iter().map(|&x| x * x).sum::<f32>())
            .sum::<f32>();

        let biases_norm: f32 = self.biases_update.iter()
            .map(|arr| arr.iter().map(|&x| x * x).sum::<f32>())
            .sum::<f32>();

        (hidden_norm + biases_norm).sqrt()
    }

    pub fn clear(&mut self) {
        self.hidden_update.iter_mut().for_each(|arr| {
            arr.map_inplace(|a| {
                *a = 0.0;
            });
        });

        self.biases_update.iter_mut().for_each(|arr| {
            arr.map_inplace(|a| {
                *a = 0.0;
            });
        });
    }
}