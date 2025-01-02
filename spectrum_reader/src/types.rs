// builtin 

// external
use ndarray::{Array, Array1, Array2, ArrayView2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

// internal
extern crate midi_encoder;
extern crate audio_to_spectrum;
use midi_encoder::types::MIDIEncoding;
use audio_to_spectrum::spectrograph::Spectrograph;


pub trait Transcribe {
    fn translate(&self, spectrum: Vec<Vec<f32>>) -> Vec<f32>;
}

pub trait Trainable {
    fn process(&mut self, input: Spectrograph, output: MIDIEncoding);

    fn record(&mut self, input: Vec<Vec<f32>>, output: Vec<f32>);
}

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

pub struct ParameterConfig {
    min_weight: f32,
    max_weight: f32,
    min_bias: f32,
    max_bias: f32,
}

impl ParameterConfig {
    pub fn new(min_weight: f32, max_weight: f32, min_bias: f32, max_bias: f32) -> ParameterConfig{
        ParameterConfig { 
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