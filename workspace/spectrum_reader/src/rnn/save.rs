// builtin
use std::fs;

// external

use super::Activation;
// internal
use super::RNN;
use super::types::*;


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


pub fn save_to_file(rnn: &RNN, file_path: &str) {
    let mut file_content: String = format!("{}", rnn.layers);
    file_content.push('\n');

    file_content.push_str(&rnn.input_size.to_string());
    file_content.push('\n');

    file_content.push_str(&rnn.output_size.to_string());
    file_content.push('\n');

    file_content.push_str(&rnn
            .units_by_layer
            .iter()
            .map(|u| u.to_string())
            .collect::<Vec<String>>()
            .join(","),
    );
    file_content.push('\n');


    for hidden_weight in rnn.hidden_weights.iter() {
        file_content.push_str(&hidden_weight.to_string());
        file_content.push('\n');
    }

    for recurrence_weight in rnn.recurrence_weights.iter() {
        file_content.push_str(&recurrence_weight.to_string());
        file_content.push('\n');
    }

    for bias in rnn.biases.iter() {
        file_content.push_str(&bias.to_string());
        file_content.push('\n');
    }

    file_content.push_str(rnn.hidden_activation.name());
    file_content.push('\n');

    file_content.push_str(rnn.end_activation.name());
    file_content.push('\n');

    fs::write(file_path, file_content).expect("Failed to write to file");
}
