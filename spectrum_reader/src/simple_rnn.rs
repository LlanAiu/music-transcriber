// builtin
use std::fs;

// external

// internal
use crate::types::{Bias, Weight};

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
}

impl RNN {
    pub fn new(
        layers: usize,
        input_size: usize,
        output_size: usize,
        units_by_layer: Vec<usize>,
    ) -> RNN {
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

            let hidden_weight: Weight = Weight::random((dim1, dim2), -0.3, 0.3);
            hidden_weights.push(hidden_weight);

            if i != layers {
                let recurrence_weight: Weight = Weight::random((dim2, dim2), -0.3, 0.3);
                recurrence_weights.push(recurrence_weight);
            }

            let bias: Bias = Bias::random(dim2, -0.3, 0.3);
            biases.push(bias);
        }

        RNN {
            layers,
            input_size,
            output_size,
            units_by_layer,
            hidden_weights,
            recurrence_weights,
            biases
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

        RNN {
            layers,
            input_size,
            output_size,
            units_by_layer,
            hidden_weights,
            recurrence_weights,
            biases
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

        fs::write(file_path, file_content).expect("Failed to write to file");
    }

    // pub fn predict(seq: Vec<Array2<f32>>) -> Vec<Vec<f32>> {
    //     let mut output_seq: Vec<Vec<f32>> = Vec::new();
    //     for arr in seq {
    //         let input_dim = arr.dim().0 - 1;
    //         let weights = arr.slice(s![1..input_dim, ..]);
    //         let biases = arr.slice(s![0..1, ..]);

    //     }

    //     output_seq
    // }

    // fn feedforward(v: Array2<f32>) -> Vec<f32> {

    // }
}
