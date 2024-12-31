// builtin
use std::fs;

// external
use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

// internal


struct RNN {
    // p
    layers: usize,

    // x
    input_size: usize,

    // o
    output_size: usize,

    //l1, l2, l3, ... lp
    units_by_layer: Vec<usize>,

    // n x l1 matrix
    input_weights: Array2<f32>,

    //lp x o matrix
    output_weights: Array2<f32>,

    // l(n-1) x ln matrix, p - 1 indices
    hidden_weights: Vec<Array2<f32>>,

    // ln x ln matrix, p indices
    recurrence_weights: Vec<Array2<f32>>,
}

impl RNN {
    pub fn new(
        layers: usize, 
        input_size: usize, 
        output_size: usize, 
        units_by_layer: Vec<usize>
    ) -> RNN {
        if layers < 1 || input_size < 1 || output_size < 1 {
            panic!("Cannot create RNN structure with specified parameters: {layers} layers, {input_size} input dim, {output_size} output dim");
        } else if units_by_layer.len() != layers {
            panic!("Layer size mismatch");
        }
 
        let mut hidden_weights: Vec<Array2<f32>> = Vec::with_capacity(layers - 1);
        let mut recurrence_weights: Vec<Array2<f32>> = Vec::with_capacity(layers);

        let first_layer_units: &usize = units_by_layer.get(0).expect("Failed to get index 0");
        let last_layer_units: &usize = units_by_layer.get(units_by_layer.len() - 1).expect("Failed to get last index");

        let input_weights: Array2<f32> = Array::random(
            (input_size, *first_layer_units),
            Uniform::new(-0.3, 0.3)
        );

        let output_weights: Array2<f32> = Array::random(
            (*last_layer_units, output_size),
            Uniform::new(-0.3, 0.3)
        );

        if units_by_layer.len() > 1{
            for i in 1..units_by_layer.len() {
                let dim1 = units_by_layer.get(i - 1).expect("Failed to get layer size");
                let dim2 = units_by_layer.get(i).expect("Failed to get layer size");

                let hidden_weight: Array2<f32> = Array::random(
                    (*dim1, *dim2), 
                    Uniform::new(-0.3, 0.3)
                );
                hidden_weights.push(hidden_weight);

                if i == 1 {
                    let first_recurrence_weight: Array2<f32> = Array::random(
                        (*dim1, *dim1), 
                        Uniform::new(-0.3, 0.3)
                    );
                    recurrence_weights.push(first_recurrence_weight);
                }

                let recurrence_weight: Array2<f32> = Array::random(
                    (*dim2, *dim2),
                    Uniform::new(-0.3, 0.3)
                );
                recurrence_weights.push(recurrence_weight);
            }
        }

        RNN {
            layers,
            input_size,
            output_size,
            units_by_layer,
            input_weights,
            output_weights,
            hidden_weights,
            recurrence_weights
        }
    }

    pub fn from_save(file_path: &str) -> RNN {
        let s: String = fs::read_to_string(file_path).expect("Failed to read save file");

        let mut lines = s.lines();
        let layers: usize = lines.next().unwrap().parse().expect("Failed to parse layers");
        let input_size: usize = lines.next().unwrap().parse().expect("Failed to parse input size");
        let output_size: usize = lines.next().unwrap().parse().expect("Failed to parse output size");
        let units_by_layer: Vec<usize> = lines.next().unwrap()
            .split(',')
            .map(|s| s.parse().expect("Failed to parse layer units"))
            .collect();

        let input_weights: Array2<f32> = Array::from_shape_vec(
            (input_size, units_by_layer[0]),
            lines.next().unwrap()
                .split(',')
                .map(|s| s.parse().expect("Failed to parse input weights"))
                .collect()
        ).expect("Failed to create input weights array");

        let output_weights: Array2<f32> = Array::from_shape_vec(
            (units_by_layer[layers - 1], output_size),
            lines.next().unwrap()
                .split(',')
                .map(|s| s.parse().expect("Failed to parse output weights"))
                .collect()
        ).expect("Failed to create output weights array");

        let mut hidden_weights: Vec<Array2<f32>> = Vec::with_capacity(layers - 1);
        for i in 0..(layers - 1) {
            let shape = (units_by_layer[i], units_by_layer[i + 1]);
            let data: Vec<f32> = lines.next().unwrap()
                .split(',')
                .map(|s| s.parse().expect("Failed to parse hidden weights"))
                .collect();
            hidden_weights.push(Array::from_shape_vec(shape, data).expect("Failed to create hidden weight array"));
        }

        let mut recurrence_weights: Vec<Array2<f32>> = Vec::with_capacity(layers);
        for i in 0..layers {
            let shape = (units_by_layer[i], units_by_layer[i]);
            let data: Vec<f32> = lines.next().unwrap()
                .split(',')
                .map(|s| s.parse().expect("Failed to parse recurrence weights"))
                .collect();
            recurrence_weights.push(Array::from_shape_vec(shape, data).expect("Failed to create recurrence weight array"));
        }

        RNN {
            layers,
            input_size,
            output_size,
            units_by_layer,
            input_weights,
            output_weights,
            hidden_weights,
            recurrence_weights
        }
    }

    pub fn load_to_file(&self, file_path: &str) {
        let mut file_content = format!(
            "{}\n{}\n{}\n{}\n",
            self.layers,
            self.input_size,
            self.output_size,
            self.units_by_layer.iter().map(|u| u.to_string()).collect::<Vec<String>>().join(",")
        );

        file_content.push_str(
            &self.input_weights.iter().map(|w| w.to_string()).collect::<Vec<String>>().join(",")
        );
        file_content.push('\n');

        file_content.push_str(
            &self.output_weights.iter().map(|w| w.to_string()).collect::<Vec<String>>().join(",")
        );
        file_content.push('\n');

        for hidden_weight in &self.hidden_weights {
            file_content.push_str(
                &hidden_weight.iter().map(|w| w.to_string()).collect::<Vec<String>>().join(",")
            );
            file_content.push('\n');
        }

        for recurrence_weight in &self.recurrence_weights {
            file_content.push_str(
                &recurrence_weight.iter().map(|w| w.to_string()).collect::<Vec<String>>().join(",")
            );
            file_content.push('\n');
        }

        fs::write(file_path, file_content).expect("Failed to write to file");
    }
}
