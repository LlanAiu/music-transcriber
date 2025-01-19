// builtin 

use std::mem::{replace, take};

// external
use ndarray::{Array, Array1, Array2, ArrayView2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

// internal
extern crate midi_encoder;
extern crate audio_to_spectrum;

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
    
    pub fn get_column_vector(&self) -> ArrayView2<f32> {
        self.array.view().insert_axis(Axis(1))
    }

    pub fn update(&mut self, update: &Array1<f32>) {
        self.array += update;
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

    pub fn update(&mut self, update: &Array2<f32>) {
        self.array += update;
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

pub struct WeightConfig {
    min_weight: f32,
    max_weight: f32,
    min_bias: f32,
    max_bias: f32,
}

impl WeightConfig {
    pub fn new(min_weight: f32, max_weight: f32, min_bias: f32, max_bias: f32) -> WeightConfig{
        WeightConfig { 
            min_weight,
            max_weight,
            min_bias,
            max_bias
        }
    }

    pub fn min_weight(&self) -> f32 {
        self.min_weight
    }

    pub fn max_weight(&self) -> f32 {
        self.max_weight
    }

    pub fn min_bias(&self) -> f32 {
        self.min_bias
    }

    pub fn max_bias(&self) -> f32 {
        self.max_bias
    }
}

//gotta rethink, this is ugly
pub struct Activation {
    function: Box<dyn FnMut(f32) -> f32>,
    name: String,
}

impl Activation {
    pub fn sigmoid() -> Activation {
        let function: Box<dyn FnMut(f32) -> f32> = Box::new(|x: f32| 1.0 / (1.0 + (-x).exp()));
        
        Activation {
            function,
            name: "sigmoid".to_string()
        }
    }

    pub fn relu() -> Activation {
        let function: Box<dyn FnMut(f32) -> f32> = Box::new(|x: f32| x.max(0.0));

        Activation {
            function,
            name: "relu".to_string()
        }
    }

    pub fn none() -> Activation {
        Activation::default()
    }

    pub fn from_string(s: &str) -> Activation {
        match s {
            "sigmoid" => { Activation::sigmoid() },
            "relu" => { Activation::relu() },
            _ => { Default::default() }
        }
    }

    pub fn of(&mut self, x: f32) -> f32 {
        (self.function)(x)
    }

    pub fn get_fn(&mut self) -> Box<dyn FnMut(f32) -> f32> {
        let function: Box<dyn FnMut(f32) -> f32>;
        if self.name() == "sigmoid" {
            function = Box::new(|x: f32| 1.0 / (1.0 + (-x).exp()));
        } else if self.name() == "relu" {
            function = Box::new(|x: f32| x.max(0.0));
        } else {
            function = Box::new(|x: f32| x);
        }
        replace(&mut self.function, function)
    }

    pub fn get_deriv(&self) -> Box<dyn FnMut(f32) -> f32> {
        let function: Box<dyn FnMut(f32) -> f32>; 
        if self.name() == "sigmoid" {
            function = Box::new(|x: f32| ((-x).exp()) / ((1.0 + (-x).exp()).powi(2)));
        } else if self.name() == "relu" {
            function = Box::new(|x: f32| {
                if x > 0.0 {
                    return 1.0;
                } else {
                    return 0.0;
                }
            });
        } else {
            function = Box::new(|x: f32| 1.0);
        }
        function
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Default for Activation {
    fn default() -> Self {
        Self { 
            function: Box::new(|x: f32| x),
            name: "none".to_string()
        }
    }
}

pub struct ActivationConfig {
    hidden: Activation,
    end: Activation
}

impl ActivationConfig {
    pub fn new(hidden: Activation, end: Activation) -> ActivationConfig {
        ActivationConfig { hidden, end }
    }

    pub fn get_hidden(&mut self) -> Activation {
        take(&mut self.hidden)
    }

    pub fn get_end(&mut self) -> Activation {
        take(&mut self.end)
    }
}

pub struct ParameterConfig {
    layers: usize,
    input_size: usize,
    output_size: usize,
    units_by_layer: Vec<usize>
}

impl ParameterConfig {
    pub fn new(layers: usize, input_size: usize, output_size: usize, units_by_layer: Vec<usize>) -> ParameterConfig {
        ParameterConfig {
            layers,
            input_size,
            output_size,
            units_by_layer
        }
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn units_by_layer(&mut self) -> Vec<usize> {
        take(&mut self.units_by_layer)
    }
}

#[derive(Debug)]
pub struct Update {
    batch_count: usize,
    max_batch_size: usize,
    hidden_update: Vec<Array2<f32>>,
    recurrence_update: Vec<Array2<f32>>,
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
        let mut recurrence_update: Vec<Array2<f32>> = Vec::with_capacity(layers);
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

            let hidden: Array2<f32> = Array2::ones((dim1, dim2));
            hidden_update.push(hidden);

            if i != layers {
                let recurrence: Array2<f32> = Array2::ones((dim2, dim2));
                recurrence_update.push(recurrence);
            }

            let bias: Array1<f32> = Array1::ones(dim2);
            biases_update.push(bias);
        }

        Update {
            batch_count: 0,
            max_batch_size,
            hidden_update,
            recurrence_update,
            biases_update
        }
    }

    pub fn should_update(&self) -> bool {
        self.batch_count >= self.max_batch_size
    }

    pub fn combine_update(&mut self, hidden: Vec<Array2<f32>>, recurrence: Vec<Array2<f32>>, biases: Vec<Array1<f32>>) {
        self.batch_count += 1;

        if hidden.len() != self.hidden_update.len() || 
            recurrence.len() != self.recurrence_update.len() ||
            biases.len() != self.biases_update.len() 
        {
            panic!("Dimension mismatch between existing and new updates");
        }

        for (i, update) in hidden.iter().enumerate() {
            self.hidden_update[i] += update;
        }

        for (i, update) in recurrence.iter().enumerate() {
            self.recurrence_update[i] += update;
        }

        for (i, update) in biases.iter().enumerate() {
            self.biases_update[i] += update;
        }
    }

    pub fn get_hidden_update(&self) -> &Vec<Array2<f32>> {
        &self.hidden_update
    }

    pub fn get_recurrence_update(&self) -> &Vec<Array2<f32>> {
        &self.recurrence_update
    }

    pub fn get_biases_update(&self) -> &Vec<Array1<f32>> {
        &self.biases_update
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
        
        self.recurrence_update.iter_mut().for_each(|arr| {
            arr.map_inplace(|a| {
                *a = 0.0;
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_test() {
        let units_by_layer: Vec<usize> = vec![5];
        let mut update: Update = Update::new(10, 2, &units_by_layer, 10);
        println!("{:?}", update);
        update.clear();

        println!("{:?}", update);
    }
}