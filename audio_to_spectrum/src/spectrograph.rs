// builtin

// external
use plotters::prelude::*;
use cqt_rs::{CQTParams, Cqt};

// internal
use crate::pcm::PCMBuffer;


pub fn pcm_to_spectrograph(pcm: PCMBuffer) -> Spectrograph {
    let params: CQTParams = CQTParams::new(
        27.5, // Min frequency
        4200.0, // Max frequency
        12, // Number of bins
        44100, // Sampling rate
        2048 // Window length
    ).expect("Error creating CQTParams");

    let cqt: Cqt = Cqt::new(params);

    let hop_size: usize = 512;

    let cqt_features = cqt.process(&pcm.samples, hop_size)
        .expect("Error computing CQT");

    let spectrograph: Vec<Vec<f32>> = cqt_features.outer_iter()
        .map(|timestep| timestep.iter().cloned().collect())
        .collect();

    Spectrograph { 
        graph: spectrograph, 
        frequency_ratio: 1.0595 
    }
}

pub struct Spectrograph {
    graph: Vec<Vec<f32>>,
    frequency_ratio: f32,
}

impl Spectrograph {
    pub fn find_max_frequency(&self) -> Vec<(usize, f32, f32)> {
        self.graph
            .iter()
            .map(|timestep| {
                let (index, max) = find_max(timestep);
                let frequency = self.frequency_ratio.powi(index as i32);
                (index, max, frequency)
            })
            .collect()
    }

    pub fn graph_ref(&self) -> &Vec<Vec<f32>> {
        &self.graph
    }

    pub fn generate_heatmap(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_value = self.graph.iter().flatten().cloned().fold(0./0., f32::max);
        let min_value = self.graph.iter().flatten().cloned().fold(0./0., f32::min);

        let mut chart = ChartBuilder::on(&root)
            .caption("Spectrograph Heatmap", ("sans-serif", 50).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                (0..self.graph[0].len()).log_scale(),
                0..self.graph.len()
            )?;

        chart.configure_mesh().draw()?;

        for (y, row) in self.graph.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                let intensity = (value - min_value) / (max_value - min_value);
                let color = RGBColor(
                    (255.0 * intensity) as u8,
                    0,
                    0,
                );
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(x, y), (x + 1, y + 1)],
                    color.filled(),
                )))?;
            }
        }

        root.present()?;
        Ok(())
    }
}

// fn split_by_timestep<T: Clone>(vector: &Vec<T>, samples: usize) -> Vec<Vec<T>> {
//     let mut iter = vector.chunks(samples);
//     let mut split: Vec<Vec<T>> = Vec::new();

//     let mut ptr = iter.next();
//     while !ptr.is_none() {
//         let val = ptr.unwrap_or(&[]);
//         split.push(val.to_vec());
//         ptr = iter.next();
//     }

//     split
// }

fn find_max(vector: &Vec<f32>) -> (usize, f32) {
    let mut max: f32 = 0.0;
    let mut index: usize = 0;
    for (i, v) in vector.iter().enumerate() {
        if *v > max {
            max = *v;
            index = i;
        }
    }
    (index, max)
}

#[cfg(test)]
mod test {
    use super::*;

    
}
