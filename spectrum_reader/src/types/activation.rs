// builtin
use std::mem::replace;
use std::sync::{OnceLock, RwLock};
// use core::cell::OnceCell;
use std::collections::HashMap;

// external

// internal


pub fn init_registry() {
    let activation_registry: ActivationRegistry = ActivationRegistry::init();

    REGISTRY_INSTANCE.set(RwLock::new(activation_registry)).unwrap();
}


#[derive(Debug)]
pub struct ActivationRegistry {
    registry: HashMap<String, Box<dyn ActivationFunction>>
}

static REGISTRY_INSTANCE: OnceLock<RwLock<ActivationRegistry>> = OnceLock::new();

impl ActivationRegistry {

    fn init() -> ActivationRegistry {
        let hmap: HashMap<String, Box<dyn ActivationFunction>> = HashMap::new();

        ActivationRegistry {
            registry: hmap
        }
    }

    pub fn get(name: &str) -> Box<dyn ActivationFunction> {
        let guard = REGISTRY_INSTANCE.get()
            .expect("Tried to use registry prior to initialization")
            .read().expect("Failed to acquire read lock on registry");

        guard.registry.get(name).map(|f| {
            f.copy()
        }).expect("Failed to fetch activation function from registry")
    }

    pub fn register(name: &str, func: Box<dyn ActivationFunction>) {
        let mut guard = REGISTRY_INSTANCE.get()
            .expect("Tried to use registry prior to initialization")
            .write().expect("Failed to acquire write lock on registry");

        guard.registry.insert(name.to_string(), func);
    }
}


pub trait ActivationFunction: Send + Sync + Debug {
    fn of(&self, x: f32) -> f32;
    fn d_of(&self, x: f32) -> f32;
    fn name(&self) -> String;
    fn copy(&self) -> Box<dyn ActivationFunction>;
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