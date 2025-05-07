// builtin
use std::error::Error;

// external
use ndarray::{Array1, Array2, Axis};

// internal
mod save;
use crate::networks::activation::Activation;
use crate::networks::computations::*;
use crate::networks::configs::{ActivationConfig, ParameterConfig, WeightConfig};
use crate::networks::types::{Bias, Update, Weight};
use save::{from_save, save_to_file};

const LEARNING_RATE: f32 = 0.001;

pub struct NN {
    // p
    layers: usize,

    input_size: usize,          // i
    output_size: usize,         // o
    units_by_layer: Vec<usize>, //l0, l1, .. l(p-1)

    // i x l0, l0 x l1, l1 x l2, ... l(p-1) x o; p + 1 indices
    hidden_weights: Vec<Weight>,

    // l0, l1, ... l(p-1), o; p + 1 indices
    biases: Vec<Bias>,

    hidden_activation: Activation,
    end_activation: Activation,
}

impl NN {
    pub fn new(
        params: &mut ParameterConfig,
        weights: WeightConfig,
        activations: &mut ActivationConfig,
    ) -> NN {
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
        let mut biases: Vec<Bias> = Vec::with_capacity(layers + 1);

        for i in 0..=layers {
            let dim1: usize;
            let dim2: usize;

            if i == 0 {
                dim1 = input_size;
                dim2 = *units_by_layer
                    .get(i)
                    .expect("Failed to get layer dimension");
            } else if i == layers {
                dim1 = *units_by_layer
                    .get(i - 1)
                    .expect("Failed to get layer dimension");
                dim2 = output_size;
            } else {
                dim1 = *units_by_layer
                    .get(i - 1)
                    .expect("Failed to get layer dimension");
                dim2 = *units_by_layer
                    .get(i)
                    .expect("Failed to get layer dimension");
            }

            let hidden_weight: Weight =
                Weight::random((dim1, dim2), weights.min_weight(), weights.max_weight());
            hidden_weights.push(hidden_weight);

            let bias: Bias = Bias::random(dim2, weights.min_bias(), weights.max_bias());
            biases.push(bias);
        }

        NN {
            layers,
            input_size,
            output_size,
            units_by_layer,
            hidden_weights,
            biases,
            hidden_activation: activations.get_hidden(),
            end_activation: activations.get_end(),
        }
    }

    pub fn from_save(file_path: &str) -> Result<NN, Box<dyn Error>> {
        from_save(file_path)
    }

    pub fn save_to_file(&self, file_path: &str) {
        save_to_file(self, file_path);
    }

    pub fn input_dim(&self) -> usize {
        self.input_size
    }

    pub fn output_dim(&self) -> usize {
        self.output_size
    }

    pub fn predict(&mut self, seq: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut output_seq: Vec<Vec<f32>> = Vec::new();

        for arr in seq {
            let (output, _activations, _raw) = self.feedforward(arr);
            output_seq.push(output);
        }

        output_seq
    }

    pub fn predict_first_layer(&mut self, seq: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut output_seq: Vec<Vec<f32>> = Vec::new();

        for arr in seq {
            let output: Vec<f32> = self.ff_single_layer(arr);
            output_seq.push(output);
        }

        output_seq
    }

    pub fn first_layer_input(&mut self, seq: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut output_seq: Vec<Vec<f32>> = Vec::new();

        for arr in seq {
            let output: Vec<f32> = self.ff_from_first(arr);
            output_seq.push(output);
        }

        output_seq
    }

    fn feedforward(&mut self, v: Vec<f32>) -> (Vec<f32>, Vec<Array2<f32>>, Vec<Array2<f32>>) {
        if v.len() != self.input_size {
            panic!(
                "Invalid input size of {}, expected {}",
                v.len(),
                self.input_size
            );
        }

        let mut arr: Array2<f32> = Array1::from_vec(v).insert_axis(Axis(0));
        let mut activations: Vec<Array2<f32>> = Vec::with_capacity(self.layers + 1);
        let mut raw_nodes: Vec<Array2<f32>> = Vec::with_capacity(self.layers + 2);

        activations.push(arr.clone());
        raw_nodes.push(arr.clone());

        for i in 0..=self.layers {
            let hidden_weight: &Weight = self.hidden_weights.get(i).expect("Failed to get weight");
            arr = arr.dot(&hidden_weight.get_weight_matrix());

            let bias: &Bias = self.biases.get(i).expect("Failed to get bias");
            arr = arr + bias.get_row_vector();

            raw_nodes.push(arr.clone());

            if i < self.layers {
                arr.mapv_inplace(|x| self.hidden_activation.of(x));
                activations.push(arr.clone());
            }
        }

        arr.mapv_inplace(|x| self.end_activation.of(x));

        let output: Vec<f32> = arr.remove_axis(Axis(0)).to_vec();

        /*
        Quick note before I get tripped up again:
        - output - last output layer only
        - activations - input + activation wrapped node values up to (but not including) final output
        - raw_nodes - input + pre-activation node values for all hidden layers + output
         */
        (output, activations, raw_nodes)
    }

    fn ff_single_layer(&mut self, v: Vec<f32>) -> Vec<f32> {
        if v.len() != self.input_size {
            panic!("Invalid input size!");
        }

        let mut arr: Array2<f32> = Array1::from_vec(v).insert_axis(Axis(0));

        let hidden_weight: &Weight = self
            .hidden_weights
            .get(0)
            .expect("Failed to get hidden weight");
        arr = arr.dot(&hidden_weight.get_weight_matrix());

        let bias: &Bias = self.biases.get(0).expect("Failed to get bias");
        arr = arr + bias.get_row_vector();

        arr.mapv_inplace(|x| self.hidden_activation.of(x));

        arr.remove_axis(Axis(0)).to_vec()
    }

    fn ff_from_first(&mut self, v: Vec<f32>) -> Vec<f32> {
        let expected_input_size: usize = *(self.units_by_layer.get(0).unwrap_or(&0));
        if v.len() != expected_input_size {
            panic!(
                "Invalid input size! Expected {} but got {}",
                expected_input_size,
                v.len()
            );
        }

        let mut arr: Array2<f32> = Array1::from_vec(v).insert_axis(Axis(0));

        for i in 1..=self.layers {
            let hidden_weight: &Weight = self.hidden_weights.get(i).expect("Failed to get weight");
            arr = arr.dot(&hidden_weight.get_weight_matrix());

            let bias: &Bias = self.biases.get(i).expect("Failed to get bias");
            arr = arr + bias.get_row_vector();

            if i < self.layers {
                arr.mapv_inplace(|x| self.hidden_activation.of(x));
            }
        }

        arr.mapv_inplace(|x| self.end_activation.of(x));

        arr.remove_axis(Axis(0)).to_vec()
    }

    pub fn predict_and_update(&mut self, seq: Vec<Vec<f32>>, ans: &Vec<Vec<f32>>, batch: usize) {
        let mut update: Update = Update::new(
            self.input_size,
            self.output_size,
            &self.units_by_layer,
            batch,
        );

        for (i, arr) in seq.into_iter().enumerate() {
            let (output, activations, raw) = self.feedforward(arr);
            let answer = ans.get(i).expect("Failed to get desired output vector");
            self.add_update(&mut update, output, answer, &activations, &raw);

            if update.should_update() {
                self.process_update(&mut update, LEARNING_RATE);
            }
        }
    }

    fn add_update(
        &self,
        grad: &mut Update,
        output: Vec<f32>,
        answer: &Vec<f32>,
        act: &Vec<Array2<f32>>,
        raw: &Vec<Array2<f32>>,
    ) {
        let mut hidden_grads: Vec<Array2<f32>> = Vec::new();
        let mut bias_grads: Vec<Array1<f32>> = Vec::new();

        let raw_outputs: &Array2<f32> =
            raw.get(self.layers + 1).expect("Failed to get raw outputs");
        let mut prev_grad: Array1<f32> = self.get_output_grad(&output, answer, raw_outputs);

        for i in (0..=self.layers).rev() {
            let layer_activation: &Array2<f32> = act.get(i).expect("Failed to get activations");
            let raw_input: &Array2<f32> = raw.get(i).expect("Failed to get raw outputs");

            let hidden: Array2<f32> = self.get_hidden_grad(i, &prev_grad, layer_activation);
            let bias: Array1<f32> = self.get_bias_grad(i, &prev_grad);

            hidden_grads.insert(0, hidden);
            bias_grads.insert(0, bias);

            prev_grad = self.get_backpropogated_grad(i, prev_grad, raw_input);
        }

        grad.combine_update(hidden_grads, bias_grads);
    }

    fn get_output_grad(
        &self,
        output: &Vec<f32>,
        answer: &Vec<f32>,
        raw: &Array2<f32>,
    ) -> Array1<f32> {
        compute_output_grad(output, answer, raw, &self.hidden_activation)
    }

    fn get_hidden_grad(
        &self,
        layer: usize,
        prev_grad: &Array1<f32>,
        input_act: &Array2<f32>,
    ) -> Array2<f32> {
        let dim: (usize, usize) = self.hidden_weights[layer].dim();

        compute_hidden_grad(dim, prev_grad, input_act)
    }

    fn get_bias_grad(&self, layer: usize, prev_grad: &Array1<f32>) -> Array1<f32> {
        let expected_dim: usize;
        if layer == self.layers {
            expected_dim = self.output_size;
        } else {
            expected_dim = self.units_by_layer[layer];
        }

        if prev_grad.dim() != expected_dim {
            panic!("Mismatched bias/gradient vector dimensions!");
        }

        compute_bias_grad(prev_grad)
    }

    fn get_backpropogated_grad(
        &self,
        layer: usize,
        prev_grad: Array1<f32>,
        raw_input: &Array2<f32>,
    ) -> Array1<f32> {
        compute_backpropogated_grad(
            &self.hidden_weights[layer],
            &self.hidden_activation,
            prev_grad,
            raw_input,
        )
    }

    fn process_update(&mut self, grad: &mut Update, alpha: f32) {
        println!("Updating weights with norm: {}", grad.get_norm());

        let hidden_updates: &Vec<Array2<f32>> = grad.get_hidden_update();
        let bias_updates: &Vec<Array1<f32>> = grad.get_biases_update();

        for i in 0..=self.layers {
            let hidden_update: &Array2<f32> = hidden_updates
                .get(i)
                .expect("Failed to get hidden weight update");
            self.hidden_weights[i].update(hidden_update, alpha);

            let bias_update: &Array1<f32> = bias_updates.get(i).expect("Failed to get bias update");
            self.biases[i].update(bias_update, alpha);
        }

        grad.clear();
    }
}

#[cfg(test)]
mod tests {
    use crate::networks::activation::init_registry;

    use super::*;

    #[test]
    fn update_test() {
        init_registry();
        let mut params: ParameterConfig = ParameterConfig::new(2, 2, 2, vec![3, 3]);
        let weights: WeightConfig = WeightConfig::new(0.999, 1.0, -0.01, 0.01);
        let mut activations: ActivationConfig =
            ActivationConfig::new(Activation::relu(), Activation::none());
        let mut nn: NN = NN::new(&mut params, weights, &mut activations);

        let mut update: Update = Update::new(
            params.input_size(),
            params.output_size(),
            &nn.units_by_layer,
            4,
        );

        let sample: Vec<f32> = vec![0.0, 0.0];

        let (output, activations, raw_nodes) = nn.feedforward(sample);

        println!("Finished feedforward prediction");

        let answer: Vec<f32> = vec![1.0, 1.0];

        nn.add_update(&mut update, output, &answer, &activations, &raw_nodes);

        println!("{:?}", update);
    }
}
