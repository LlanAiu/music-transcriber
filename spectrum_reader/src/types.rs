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