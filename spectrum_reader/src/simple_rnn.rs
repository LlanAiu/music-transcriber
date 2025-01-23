// builtin
use std::fs;

// external
use ndarray::{Array1, Array2, ArrayView2, Axis};

// internal
use crate::types::{Activation, ActivationConfig, Bias, ParameterConfig, Update, Weight, WeightConfig};

const LEARNING_RATE: f32 = 0.001;

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
            let (output, activations, _raw) = self.feedforward(arr, &prev_activations);
            prev_activations = activations;
            output_seq.push(output);
        }

        output_seq
    }

    fn feedforward(&mut self, v: Vec<f32>, prev: &Vec<Array2<f32>>) -> (Vec<f32>, Vec<Array2<f32>>, Vec<Array2<f32>>) {
        if v.len() != self.input_size {
            panic!("Invalid input size");
        } 

        let mut arr: Array2<f32> = Array1::from_vec(v).insert_axis(Axis(0));
        let mut activations: Vec<Array2<f32>> = Vec::with_capacity(self.layers + 1);
        let mut raw_nodes: Vec<Array2<f32>> = Vec::with_capacity(self.layers + 2);

        activations.push(arr.clone());
        raw_nodes.push(arr.clone());

        for i in 0..=self.layers {
            let hidden_weight: &Weight = self.hidden_weights.get(i).expect("Failed to get weight");
            arr = arr.dot(&hidden_weight.get_weight_matrix());

            if i < self.layers {
                let previous: Option<&Array2<f32>> = prev.get(i + 1);

                if previous.is_some() {
                    let recurrent_weight: &Weight = self.recurrence_weights.get(i).expect("Failed to get weights");
                    let val: Array2<f32> = previous.expect("Failed to get previous activations")
                        .dot(&recurrent_weight.get_weight_matrix());
                    arr = arr + &val;
                }
            }

            let bias: &Bias = self.biases.get(i).expect("Failed to get bias");
            arr = arr + bias.get_row_vector();

            raw_nodes.push(arr.clone());
            
            if i < self.layers {
                arr.mapv_inplace(self.hidden_activation.get_fn());
                activations.push(arr.clone());
            }
        }

        arr.mapv_inplace(self.end_activation.get_fn());

        let output: Vec<f32> = arr.remove_axis(Axis(0)).to_vec();
        
        /*
        Quick note before I get tripped up again:
        - output - last output layer only
        - activations - input + activation wrapped node values up to (but not including) final output
        - raw_nodes - input + pre-activation node values for all hidden layers + output
         */
        (output, activations, raw_nodes)
    }

    pub fn predict_and_update(&mut self, seq: Vec<Vec<f32>>, ans: Vec<Vec<f32>>, batch: usize) {
        let mut prev_activations: Vec<Array2<f32>> = Vec::new();
        let mut update: Update = Update::new(self.input_size, self.output_size, &self.units_by_layer, batch);

        for (i, arr) in seq.into_iter().enumerate() {
            let (output, activations, raw) = self.feedforward(arr, &prev_activations);
            let answer = ans.get(i).expect("Failed to get desired output vector");
            self.add_update(&mut update, output, answer, &activations, &prev_activations, &raw);

            if update.should_update() {
                self.process_update(&mut update, LEARNING_RATE);
            }

            prev_activations = activations;
        }
    }

    fn add_update(&self,
        grad: &mut Update,
        output: Vec<f32>, 
        answer: &Vec<f32>, 
        act: &Vec<Array2<f32>>, 
        prev_act: &Vec<Array2<f32>>,
        raw: &Vec<Array2<f32>>
    ) {
        let mut hidden_grads: Vec<Array2<f32>> = Vec::new();
        let mut recurrence_grads: Vec<Array2<f32>> = Vec::new();
        let mut bias_grads: Vec<Array1<f32>> = Vec::new();

        let raw_outputs: &Array2<f32> = raw.get(self.layers + 1)
            .expect("Failed to get raw outputs");
        let mut prev_grad: Array1<f32> = self.get_output_grad(&output, answer, raw_outputs);

        for i in (0..=self.layers).rev() {
            let layer_activation: &Array2<f32> = act.get(i).expect("Failed to get activations");
            let raw_input: &Array2<f32> = raw.get(i).expect("Failed to get raw outputs");

            let hidden: Array2<f32> = self.compute_hidden_grad(i, &prev_grad, layer_activation);
            let bias: Array1<f32> = self.compute_bias_grad(i, &prev_grad);

            hidden_grads.insert(0, hidden);
            bias_grads.insert(0, bias);

            if i < self.layers {
                let prev_activation: Option<&Array2<f32>> = prev_act.get(i + 1);
                let recurrence: Array2<f32> = self.compute_recurrence_grad(i, &prev_grad, prev_activation);
                recurrence_grads.insert(0, recurrence);
            }

            prev_grad = self.backpropogate_grad(i, prev_grad, raw_input);
        }

        grad.combine_update(hidden_grads, recurrence_grads, bias_grads);
    }

    fn get_output_grad(&self, output: &Vec<f32>, answer: &Vec<f32>, raw: &Array2<f32>) -> Array1<f32> {
        let mut loss_vec: Vec<f32> = Vec::new();
        for (i, out) in output.iter().enumerate() {
            let expected: &f32 = answer.get(i).expect("Failed to get vector value");
            let residual: f32 = expected - out;
            loss_vec.push(residual);
        }

        let loss: Array1<f32> = Array1::from_vec(loss_vec);
        let mut scale: Array1<f32> = raw.clone().remove_axis(Axis(0));
        scale.mapv_inplace(self.end_activation.get_deriv());
        scale *= -1.0;

        loss * scale
    }

    fn compute_hidden_grad(&self, layer: usize, prev_grad: &Array1<f32>, input_act: &Array2<f32>) -> Array2<f32> {
        let dim: (usize, usize) = self.hidden_weights[layer].dim();

        let grad_matrix: ArrayView2<f32> = prev_grad.view().insert_axis(Axis(0));
        let grads: ArrayView2<f32> = grad_matrix.broadcast(dim)
            .expect("Failed to broadcast gradient vector");

        let mut hidden_update: Array2<f32> = input_act.view().reversed_axes().broadcast(dim)
            .expect("Failed to broadcast input vector").to_owned();

        hidden_update = hidden_update * grads;
        
        hidden_update
    }

    fn compute_bias_grad(&self, layer: usize, prev_grad: &Array1<f32>) -> Array1<f32> {
        let expected_dim: usize;
        if layer == self.layers {
            expected_dim = self.output_size;
        } else {
            expected_dim = self.units_by_layer[layer];
        }

        if prev_grad.dim() != expected_dim{
            panic!("Mismatched bias/gradient vector dimensions!");
        }

        prev_grad.clone()
    }

    fn compute_recurrence_grad(&self, layer: usize, prev_grad: &Array1<f32>, prev_act: Option<&Array2<f32>>) -> Array2<f32> {
        if layer >= self.layers {
            panic!("Invalid layer index!");
        }

        let layer_dim: usize = self.units_by_layer[layer];
        let dim: (usize, usize) = (layer_dim, layer_dim);

        if prev_act.is_some() {
            let activation: &Array2<f32> = prev_act.expect("Failed to get previous activations");
            let grad_matrix: ArrayView2<f32> = prev_grad.view().insert_axis(Axis(0));
            let grads: ArrayView2<f32> = grad_matrix.broadcast(dim)
                .expect("Failed to broadcast gradient vector");

            let mut recurrence_update: Array2<f32> = activation.view().reversed_axes().broadcast(dim)
                .expect("Failed to broadcast previous inputs").to_owned();

            recurrence_update = recurrence_update * grads;

            return recurrence_update;
        } else {
            return Array2::zeros(dim);
        }
    }

    fn backpropogate_grad(
        &self, 
        layer: usize, 
        prev_grad: Array1<f32>, 
        raw_input: &Array2<f32>,
    ) -> Array1<f32> {
    
        let mut input_vec: Array1<f32> = raw_input.view().remove_axis(Axis(0)).to_owned();
        input_vec.mapv_inplace(self.hidden_activation.get_deriv());

        let weights: ArrayView2<f32> = self.hidden_weights[layer].get_weight_matrix();
        let grads: Array2<f32> = prev_grad.insert_axis(Axis(1));

        let result = weights.dot(&grads).remove_axis(Axis(1));

        result * input_vec
    }

    fn process_update(&mut self, grad: &mut Update, alpha: f32) {
        let hidden_updates: &Vec<Array2<f32>> = grad.get_hidden_update();
        let bias_updates: &Vec<Array1<f32>> = grad.get_biases_update();
        let recurrence_updates: &Vec<Array2<f32>> = grad.get_recurrence_update();

        for i in 0..=self.layers {
            let hidden_update: &Array2<f32> = hidden_updates.get(i).expect("Failed to get hidden weight update");
            self.hidden_weights[i].update(hidden_update, alpha);

            let bias_update: &Array1<f32> = bias_updates.get(i).expect("Failed to get bias update");
            self.biases[i].update(bias_update, alpha);

            if i < self.layers {
                let recurrence_update: &Array2<f32> = recurrence_updates.get(i).expect("Failed to get recurrence update");
                self.recurrence_weights[i].update(recurrence_update, alpha);
            }
        }

        grad.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_gradient_test(){
        let mut params: ParameterConfig = ParameterConfig::new(1, 3, 2, vec![4]);
        let weights: WeightConfig = WeightConfig::new(0.999, 1.0, -0.01, 0.01);
        let mut activations: ActivationConfig = ActivationConfig::new(Activation::none(), Activation::none());
        let rnn: RNN = RNN::new(&mut params, weights, &mut activations);

        let prev_grad_vec: Vec<f32> = vec![1.0, 2.0];
        let prev_grad: Array1<f32> = Array1::from_vec(prev_grad_vec);

        let input_act_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input_act: Array2<f32> = Array1::from_vec(input_act_vec).insert_axis(Axis(0));

        let gradient = rnn.compute_hidden_grad(1, &prev_grad, &input_act);
    

        let ans_vec: Vec<f32> = vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0];
        let ans: Array2<f32> = Array2::from_shape_vec((4, 2), ans_vec).expect("Failed to create ans matrix");

        println!("{:?}", gradient);
        assert_eq!(gradient, ans);
    }

    #[test]
    fn bias_gradient_test(){
        let mut params: ParameterConfig = ParameterConfig::new(1, 3, 2, vec![4]);
        let weights: WeightConfig = WeightConfig::new(0.999, 1.0, -0.01, 0.01);
        let mut activations: ActivationConfig = ActivationConfig::new(Activation::none(), Activation::none());
        let rnn: RNN = RNN::new(&mut params, weights, &mut activations);

        let prev_grad_vec: Vec<f32> = vec![1.0, 2.0];
        let prev_grad: Array1<f32> = Array1::from_vec(prev_grad_vec);

        let gradient = rnn.compute_bias_grad(1, &prev_grad);

        let ans_vec: Vec<f32> = vec![1.0, 2.0];
        let ans: Array1<f32> = Array1::from_vec(ans_vec);

        println!("{:?}", gradient);
        assert_eq!(gradient, ans);
    }

    #[test]
    fn recurrence_gradient_test() {
        let mut params: ParameterConfig = ParameterConfig::new(1, 2, 2, vec![3]);
        let weights: WeightConfig = WeightConfig::new(0.999, 1.0, -0.01, 0.01);
        let mut activations: ActivationConfig = ActivationConfig::new(Activation::none(), Activation::none());
        let rnn: RNN = RNN::new(&mut params, weights, &mut activations);

        let prev_grad_vec: Vec<f32> = vec![1.0, 1.0, 1.0];
        let prev_grad: Array1<f32> = Array1::from_vec(prev_grad_vec);

        let prev_act_vec: Vec<f32> = vec![1.0, 2.0, 3.0];
        let prev_act: Array2<f32> = Array1::from_vec(prev_act_vec).insert_axis(Axis(0));

        let act_option: Option<&Array2<f32>> = Some(&prev_act);

        let gradient: Array2<f32> = rnn.compute_recurrence_grad(0, &prev_grad, act_option);

        let ans_vec: Vec<f32> = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let ans: Array2<f32> = Array2::from_shape_vec((3, 3), ans_vec).expect("Failed to create ans matrix");

        println!("{:?}", gradient);
        assert_eq!(gradient, ans);
    }

    #[test]
    fn backprop_test() {
        let mut params: ParameterConfig = ParameterConfig::new(1, 2, 2, vec![3]);
        let weights: WeightConfig = WeightConfig::new(0.999, 1.0, -0.01, 0.01);
        let mut activations: ActivationConfig = ActivationConfig::new(Activation::relu(), Activation::relu());
        let rnn: RNN = RNN::new(&mut params, weights, &mut activations);

        let prev_grad_vec: Vec<f32> = vec![2.0, 2.0];
        let prev_grad: Array1<f32> = Array1::from_vec(prev_grad_vec);

        let raw_input_vec: Vec<f32> = vec![1.0, -1.0, 1.0];
        let raw_input: Array2<f32> = Array1::from_vec(raw_input_vec).insert_axis(Axis(0));

        let mut grad: Array1<f32> = rnn.backpropogate_grad(1, prev_grad, &raw_input);

        grad.mapv_inplace(|x| x.round());

        let ans_vec: Vec<f32> = vec![4.0, 0.0, 4.0];
        let ans: Array1<f32> = Array1::from_vec(ans_vec);

        println!("{:?}", grad);

        assert_eq!(grad, ans);
    }

    #[test]
    fn output_grad_test() {
        let mut params: ParameterConfig = ParameterConfig::new(1, 2, 2, vec![3]);
        let weights: WeightConfig = WeightConfig::new(0.999, 1.0, -0.01, 0.01);
        let mut activations: ActivationConfig = ActivationConfig::new(Activation::relu(), Activation::relu());
        let rnn: RNN = RNN::new(&mut params, weights, &mut activations);

        let output: Vec<f32> = vec![0.0, 0.0];
        let answer: Vec<f32> = vec![4.0, 2.0];

        let raw_vec: Vec<f32> = vec![-1.0, 1.0];
        let raw_output: Array2<f32> = Array1::from_vec(raw_vec).insert_axis(Axis(0));

        let mut grad: Array1<f32> = rnn.get_output_grad(&output, &answer, &raw_output);

        grad.mapv_inplace(|x | x.round());

        let ans_vec: Vec<f32> = vec![0.0, -2.0];
        let ans: Array1<f32> = Array1::from_vec(ans_vec);

        println!("{:?}", grad);
        assert_eq!(grad, ans);
    }

    #[test]
    fn update_test() {
        let mut params: ParameterConfig = ParameterConfig::new(2, 2, 2, vec![3, 3]);
        let weights: WeightConfig = WeightConfig::new(0.999, 1.0, -0.01, 0.01);
        let mut activations: ActivationConfig = ActivationConfig::new(Activation::relu(), Activation::none());
        let mut rnn: RNN = RNN::new(&mut params, weights, &mut activations);

        let mut update: Update = Update::new(params.input_size(), params.output_size(), &rnn.units_by_layer, 4);

        let sample: Vec<f32> = vec![0.0, 0.0];
        let ( _, prev_activations, _ ) = rnn.feedforward(sample.clone(), &Vec::new());
        
        let (
            output, 
            activations, 
            raw_nodes
        ) = rnn.feedforward(sample, &prev_activations);

        println!("Finished feedforward prediction");

        let answer: Vec<f32> = vec![1.0, 1.0];


        rnn.add_update(
            &mut update, 
            output, 
            &answer, 
            &activations, 
            &prev_activations, 
            &raw_nodes
        );

        println!("{:?}", update);
    }
}