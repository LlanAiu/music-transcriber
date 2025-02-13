// builtin 
use std::fs;
use std::error::Error;
// external

// internal
use super::NN;
use crate::networks::activation::Activation;
use crate::networks::types::{Bias, Weight};


pub fn from_save(file_path: &str) -> Result<NN, Box<dyn Error>> {
    let s: String = fs::read_to_string(file_path)?;
    let mut lines = s.lines();

    let layers: usize = lines.next().expect("Failed to get next line").parse()?;

    let input_size: usize = lines.next().expect("Failed to get next line").parse()?;
    
    let output_size: usize = lines.next().expect("Failed to get next line").parse()?;

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

    let mut biases: Vec<Bias> = Vec::with_capacity(layers + 1);
    for _i in 0..=layers {
        let data_str: &str = lines.next().expect("Failed to get next line");
        let bias: Bias = Bias::from_string(data_str);
        biases.push(bias);
    }

    let hidden_activation: Activation = Activation::from_string(lines.next().expect("Failed to get next line"));
    let end_activation: Activation = Activation::from_string(lines.next().expect("Failed to get next line"));

    Result::Ok(NN {
        layers,
        input_size,
        output_size,
        units_by_layer,
        hidden_weights,
        biases,
        hidden_activation,
        end_activation
    })
}


pub fn save_to_file(nn: &NN, file_path: &str) {
    let mut file_content: String = format!("{}", nn.layers);
    file_content.push('\n');

    file_content.push_str(&nn.input_size.to_string());
    file_content.push('\n');

    file_content.push_str(&nn.output_size.to_string());
    file_content.push('\n');

    file_content.push_str(&nn
            .units_by_layer
            .iter()
            .map(|u| u.to_string())
            .collect::<Vec<String>>()
            .join(","),
    );
    file_content.push('\n');


    for hidden_weight in nn.hidden_weights.iter() {
        file_content.push_str(&hidden_weight.to_string());
        file_content.push('\n');
    }

    for bias in nn.biases.iter() {
        file_content.push_str(&bias.to_string());
        file_content.push('\n');
    }

    file_content.push_str(nn.hidden_activation.name());
    file_content.push('\n');

    file_content.push_str(nn.end_activation.name());
    file_content.push('\n');

    println!("Writing to file...");

    fs::write(file_path, file_content).expect("Failed to write to file");
}