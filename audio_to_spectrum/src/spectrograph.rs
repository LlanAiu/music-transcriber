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

    let hop_size: usize = 512; // timestep of 512/44100 

    let cqt_features = cqt.process(&pcm.samples, hop_size)
        .expect("Error computing CQT");

    let max_value = cqt_features.iter().cloned().fold(0./0., f32::max);
    let spectrograph: Vec<Vec<f32>> = cqt_features.outer_iter()
        .map(|timestep| timestep.iter().map(|&val| val / max_value).collect())
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
                let frequency = 27.5 * self.frequency_ratio.powi(index as i32);
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

        let mut chart = ChartBuilder::on(&root)
            .caption("Spectrograph Heatmap", ("sans-serif", 50).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0..self.graph[0].len(),
                0..self.graph.len()
            )?;

        chart.configure_mesh().draw()?;

        for (y, row) in self.graph.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                let color = RGBColor(
                    (255.0 * value) as u8,
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
    use crate::pcm::{audio_to_pcm, AudioConfig};

    use super::*;

    #[test]
    fn basic_frequency_test() {
        let pcm: PCMBuffer = audio_to_pcm(AudioConfig::new("./tests/700hz_test.mp3"));
        let graph: Spectrograph = pcm_to_spectrograph(pcm);
        const TIME_PER_INDEX: f32 = 512.0 / 44100.0;

        for (i, (index, value, frequency)) in graph.find_max_frequency().iter().enumerate() {
            let time: f32 = TIME_PER_INDEX * (i as f32);
            println!("Time {time}: max value {value} @ index {index} w/ frequency {frequency}");
        }
    }
} 
