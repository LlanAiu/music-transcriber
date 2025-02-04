// builtin 
use std::mem::take;

// external

// internal
pub mod activation;
pub use activation::Activation;

pub struct ConverterConfig {
    layers: usize, 
    units_by_layer: Vec<usize>,
    batch_size: usize
}

impl ConverterConfig {
    pub fn new(layers: usize, units_by_layer: Vec<usize>, batch_size: usize) -> ConverterConfig {
        ConverterConfig {
            layers,
            units_by_layer,
            batch_size,
        }
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn units_by_layer(&mut self) -> Vec<usize> {
        take(&mut self.units_by_layer)
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
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

    pub fn units_by_layer_ref(&self) -> &Vec<usize> {
        &self.units_by_layer
    }

    pub fn units_by_layer(&mut self) -> Vec<usize> {
        take(&mut self.units_by_layer)
    }
}