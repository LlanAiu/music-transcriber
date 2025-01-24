// builtin 
use std::mem::{replace, take};

// external

// internal


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
            function = Box::new(|_x: f32| 1.0);
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