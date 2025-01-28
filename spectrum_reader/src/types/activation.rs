// builtin
use std::fmt::Debug;
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
    function: Box<dyn ActivationFunction>,
    name: String,
}

impl Activation {
    pub fn sigmoid() -> Activation {
        let function: Box<dyn ActivationFunction> = Box::new(Sigmoid);
        
        Activation {
            function,
            name: "sigmoid".to_string()
        }
    }

    pub fn relu() -> Activation {
        let function: Box<dyn ActivationFunction> = Box::new(ReLU);

        Activation {
            function,
            name: "relu".to_string()
        }
    }

    pub fn none() -> Activation {
        Activation::default()
    }

    pub fn from_function(func: Box<dyn ActivationFunction>) -> Activation {
        Activation {
            name: func.name(),
            function: func
        }
    }

    pub fn from_string(s: &str) -> Activation {
        let function: Box<dyn ActivationFunction> = ActivationRegistry::get(s);

        Activation {
            function,
            name: s.to_string()
        }
    }

    pub fn of(&self, x: f32) -> f32 {
        self.function.of(x)
    }

    pub fn deriv_of(&self, x: f32) -> f32 {
        self.function.d_of(x)
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Default for Activation {
    fn default() -> Self {
        Activation::from_function(Box::new(None))
    }
}

pub struct ReLU;

impl ActivationFunction for ReLU {
    fn of(&self, x: f32) -> f32 {
        x.max(0.0)
    }

    fn d_of(&self, x: f32) -> f32 {
        if x > 0.0 {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    fn name(&self) -> String {
        "relu".to_string()
    }

    fn copy(&self) -> Box<dyn ActivationFunction> {
        Box::new(ReLU)
    }
}

impl Debug for ReLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReLU").finish()
    }
}

pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn of(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn d_of(&self, x: f32) -> f32 {
        ((-x).exp()) / ((1.0 + (-x).exp()).powi(2))
    }

    fn name(&self) -> String {
        "sigmoid".to_string()
    }

    fn copy(&self) -> Box<dyn ActivationFunction> {
        Box::new(Sigmoid)
    }
}

impl Debug for Sigmoid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sigmoid").finish()
    }
}

pub struct None;

impl ActivationFunction for None {
    fn of(&self, x: f32) -> f32 {
        x
    }

    fn d_of(&self, x: f32) -> f32 {
        1.0
    }

    fn name(&self) -> String {
        "none".to_string()
    }

    fn copy(&self) -> Box<dyn ActivationFunction> {
        Box::new(None)
    }
}

impl Debug for None {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("None").finish()
    }
}