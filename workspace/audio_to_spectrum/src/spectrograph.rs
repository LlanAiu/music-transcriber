// builtin

use std::mem::take;

// external
use plotters::prelude::*;
use cqt_rs::{CQTParams, Cqt};
use constants::*;

// internal
use crate::pcm::PCMBuffer;

pub mod constants {
    pub const MIN_FREQ: f32 = 27.5;
    pub const MAX_FREQ: f32 = 4200.0;
    pub const BINS_PER_OCTAVE: usize = 12;
    pub const SAMPLE_RATE: usize = 44100;
    pub const WINDOW_LENGTH: usize = 2048;
    pub const HOP_SIZE: usize = 512;
    pub const TIME_STEP: f32 = (HOP_SIZE as f32) / (SAMPLE_RATE as f32);
    pub const DEADBAND: f32 = 0.01;
}

pub fn pcm_to_spectrograph(pcm: PCMBuffer) -> Spectrograph {
    let params: CQTParams = CQTParams::new(
        MIN_FREQ,
        MAX_FREQ,
        BINS_PER_OCTAVE,
        SAMPLE_RATE,
        WINDOW_LENGTH
    ).expect("Error creating CQTParams");

    let cqt: Cqt = Cqt::new(params);

    let hop_size: usize = HOP_SIZE;

    let cqt_features = cqt.process(&pcm.samples, hop_size)
        .expect("Error computing CQT");

    let max_value = cqt_features.iter().cloned().fold(0./0., f32::max);
    let spectrograph: Vec<Vec<f32>> = cqt_features.outer_iter()
        .map(|timestep| timestep.iter().map(|&val| {
            let normalized_val = val / max_value;
            if normalized_val < DEADBAND { 0.0 } else { normalized_val }
        }).collect())
        .collect();

    Spectrograph { 
        graph: spectrograph, 
        frequency_ratio: 1.0595,
        timestep_ms: TIME_STEP
    }
}

#[derive(Clone)]
pub struct Spectrograph {
    graph: Vec<Vec<f32>>,
    frequency_ratio: f32,
    timestep_ms: f32
}

impl Spectrograph {
    pub fn vector_dim(&self) -> usize {
        if self.num_timestamps() > 0 {
            self.graph[0].len()
        } else {
            0
        }
    }

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

    pub fn graph(&mut self) -> Vec<Vec<f32>> {
        take(&mut self.graph)
    }

    pub fn num_timestamps(&self) -> usize {
        self.graph.len()
    }

    pub fn get_timestep(&self) -> f32 {
        self.timestep_ms
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
