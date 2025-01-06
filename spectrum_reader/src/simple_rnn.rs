// builtin
use std::fs;

// external

use ndarray::{Array1, Array2, Axis};

// internal
use crate::types::{Activation, ActivationConfig, Bias, ParameterConfig, Update, Weight, WeightConfig};

pub struct RNN {
    // p
    layers: usize,

    input_size: usize,          // i
    output_size: usize,         // o
    units_by_layer: Vec<usize>, //l0, l1, .. l(p-1)

    // i x l0, l0 x l1, l1 x l2, ... l(p-1) x o; p + 1 indices
    hidden_weights: Vec<Weight>,

    // ln x ln matrix, p indices
    recurrence_weights: Vec<Weight>,

    // l0, l1, ... l(p-1), o; p + 1 indices
    biases: Vec<Bias>,

    hidden_activation: Activation,
    end_activation: Activation,
}

impl RNN {
    pub fn new(
        params: &mut ParameterConfig,
        weights: WeightConfig,
        activations: &mut ActivationConfig
    ) -> RNN {
        let layers: usize = params.layers();
        let input_size: usize = params.input_size();
        let output_size: usize = params.output_size();
        let units_by_layer: Vec<usize> = params.units_by_layer();

        if layers < 1 || input_size < 1 || output_size < 1 {
            panic!("Cannot create RNN structure with specified parameters: {layers} layers, {input_size} input dim, {output_size} output dim");
        } else if units_by_layer.len() != layers {
            panic!("Layer size mismatch");
        }

        let mut hidden_weights: Vec<Weight> = Vec::with_capacity(layers + 1);
        let mut recurrence_weights: Vec<Weight> = Vec::with_capacity(layers);
        let mut biases: Vec<Bias> = Vec::with_capacity(layers + 1);

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

            let hidden_weight: Weight = Weight::random((dim1, dim2), weights.min_weight(), weights.max_weight());
            hidden_weights.push(hidden_weight);

            if i != layers {
                let recurrence_weight: Weight = Weight::random((dim2, dim2), weights.min_weight(), weights.max_weight());
                recurrence_weights.push(recurrence_weight);
            }

            let bias: Bias = Bias::random(dim2, weights.min_bias(), weights.max_bias());
            biases.push(bias);
        }

        RNN {
            layers,
            input_size,
            output_size,
            units_by_layer,
            hidden_weights,
            recurrence_weights,
            biases,
            hidden_activation: activations.get_hidden(),
            end_activation: activations.get_end()
        }
    }

    pub fn from_save(file_path: &str) -> RNN {
        let s: String = fs::read_to_string(file_path).expect("Failed to read save file");
        let mut lines = s.lines();

        let layers: usize = lines.next().expect("Failed to get next line")
            .parse().expect("Failed to parse layers");

        let input_size: usize = lines.next().expect("Failed to get next line")
            .parse().expect("Failed to parse input size");
        
        let output_size: usize = lines.next().expect("Failed to get next line")
            .parse().expect("Failed to parse output size");

        let units_by_layer: Vec<usize> = lines.next().expect("Failed to get next line")
            .split(',')
            .map(|s| s.parse().expect("Failed to parse layer dimension"))
            .collect();

        let mut hidden_weights: Vec<Weight> = Vec::with_capacity(layers + 1);
        for _i in 0..=layers {
            let data_str: &str = lines.next().expect("Failed to get next line");
            let weight: Weight = Weight::from_string(data_str);
            hidden_weights.push(weight);
        }

        let mut recurrence_weights: Vec<Weight> = Vec::with_capacity(layers);
        for _i in 0..layers {
            let data_str: &str = lines.next().expect("Failed to get next line");
            let weight: Weight = Weight::from_string(data_str);
            recurrence_weights.push(weight);
        }

        let mut biases: Vec<Bias> = Vec::with_capacity(layers + 1);
        for _i in 0..=layers {
            let data_str: &str = lines.next().expect("Failed to get next line");
            let bias: Bias = Bias::from_string(data_str);
            biases.push(bias);
        }

        let hidden_activation: Activation = Activation::from_string(lines.next().expect("Failed to get next line"));
        let end_activation: Activation = Activation::from_string(lines.next().expect("Failed to get next line"));

        RNN {
            layers,
            input_size,
            output_size,
            units_by_layer,
            hidden_weights,
            recurrence_weights,
            biases,
            hidden_activation,
            end_activation
        }
    }

    pub fn save_to_file(&self, file_path: &str) {
        let mut file_content: String = format!("{}", self.layers);
        file_content.push('\n');

        file_content.push_str(&self.input_size.to_string());
        file_content.push('\n');

        file_content.push_str(&self.output_size.to_string());
        file_content.push('\n');

        file_content.push_str(&self
                .units_by_layer
                .iter()
                .map(|u| u.to_string())
                .collect::<Vec<String>>()
                .join(","),
        );
        file_content.push('\n');


        for hidden_weight in self.hidden_weights.iter() {
            file_content.push_str(&hidden_weight.to_string());
            file_content.push('\n');
        }

        for recurrence_weight in self.recurrence_weights.iter() {
            file_content.push_str(&recurrence_weight.to_string());
            file_content.push('\n');
        }

        for bias in self.biases.iter() {
            file_content.push_str(&bias.to_string());
            file_content.push('\n');
        }

        file_content.push_str(self.hidden_activation.name());
        file_content.push('\n');

        file_content.push_str(self.end_activation.name());
        file_content.push('\n');

        fs::write(file_path, file_content).expect("Failed to write to file");
    }

    pub fn predict(&mut self, seq: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut output_seq: Vec<Vec<f32>> = Vec::new();
        let mut prev_activations: Vec<Array2<f32>> = Vec::new();

        for arr in seq {
            let (output, activations) = self.feedforward(arr, &prev_activations);
            prev_activations = activations;
            output_seq.push(output);
        }

        output_seq
    }

    fn feedforward(&mut self, v: Vec<f32>, prev: &Vec<Array2<f32>>) -> (Vec<f32>, Vec<Array2<f32>>) {
        if v.len() != self.input_size {
            panic!("Invalid input size");
        } 

        let mut arr: Array2<f32> = Array1::from_vec(v).insert_axis(Axis(0));
        let mut activations: Vec<Array2<f32>> = Vec::with_capacity(self.layers);
        
        for i in 0..=self.layers {
            let hidden_weight: &Weight = self.hidden_weights.get(i).expect("Failed to get weight");
            arr = arr.dot(&hidden_weight.get_weight_matrix());

            if i < self.layers {
                let previous: Option<&Array2<f32>> = prev.get(i);

                if previous.is_some() {
                    let recurrent_weight: &Weight = self.recurrence_weights.get(i).expect("Failed to get weights");
                    let val: Array2<f32> = previous.expect("Failed to get previous activations")
                        .dot(&recurrent_weight.get_weight_matrix());
                    arr = arr + &val;
                }
            }

            let bias: &Bias = self.biases.get(i).expect("Failed to get bias");
            arr = arr + bias.get_row_vector();

            arr.mapv_inplace(self.hidden_activation.get_fn());
            
            if i < self.layers {
                activations.push(arr.clone());
            }
        }

        arr.mapv_inplace(self.end_activation.get_fn());

        let output: Vec<f32> = arr.remove_axis(Axis(0)).to_vec();
        
        (output, activations)
    }

    pub fn predict_and_update(&mut self, seq: Vec<Vec<f32>>, ans: Vec<Vec<f32>>, batch: usize) {
        let mut prev_activations: Vec<Array2<f32>> = Vec::new();
        let mut update: Update = Update::new(self.input_size, self.output_size, &self.units_by_layer, batch);

        for (i, arr) in seq.into_iter().enumerate() {
            let (output, activations) = self.feedforward(arr, &prev_activations);
            let answer = ans.get(i).expect("Failed to get desired output vector");
            self.add_update(&mut update, output, answer, &activations, &prev_activations);

            if update.should_update() {
                self.process_update(&mut update);
            }

            prev_activations = activations;
        }
    }

    fn add_update(&self,
        grad: &mut Update,
        output: Vec<f32>, 
        answer: &Vec<f32>, 
        act: &Vec<Array2<f32>>, 
        prev_act: &Vec<Array2<f32>>
    ) {
        todo!()
    }

    fn process_update(&mut self, grad: &mut Update) {
        todo!()
    }
}
