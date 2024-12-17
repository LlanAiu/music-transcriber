use rustfft::{num_complex::Complex, FftPlanner};

use crate::pcm::PCMBuffer;

pub struct PCMConverter {
    fft_len: usize,
    sample_size: usize,
}

impl PCMConverter {
    const AUDIO_RATE: usize = 44100;

    pub fn new(fft_len: usize, sample_size: usize) -> PCMConverter {
        PCMConverter {
            fft_len,
            sample_size,
        }
    }

    pub fn to_spectrograph(&self, pcm: PCMBuffer) -> Spectrograph {
        let complex_data: Vec<Complex<f32>> = pcm.samples
            .into_iter()
            .map(|sample| Complex {
                re: sample,
                im: 0.0,
            })
            .collect();

        let sample_matrix: Vec<Vec<Complex<f32>>> = split_by_timestep(&complex_data, self.sample_size);
        let mut spectrograph: Vec<Vec<f32>> = Vec::new();

        let mut planner: FftPlanner<f32> = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.fft_len);

        for mut sample in sample_matrix {
            if sample.len() < self.fft_len {
                sample.resize(self.fft_len, Complex { re: 0.0, im: 0.0 });
            }

            fft.process(&mut sample);

            let normalization_factor = (self.fft_len as f32).sqrt();
            let mut sample_graph: Vec<f32> = sample
                .iter()
                .map(|val| val.norm() / normalization_factor)
                .collect();

            sample_graph.truncate(self.fft_len / 2);

            spectrograph.push(sample_graph);
        }

        Spectrograph { graph: spectrograph, frequency_res: (PCMConverter::AUDIO_RATE as f32) / (self.fft_len as f32) }
    }

}

pub struct Spectrograph {
    graph: Vec<Vec<f32>>,
    frequency_res: f32
}

impl Spectrograph {
    pub fn find_max_frequency(&self) -> Vec<(usize, f32, f32)> {
        self.graph
            .iter()
            .map(|timestep| {
                let (index, max) = find_max(timestep);
                let frequency = index as f32 * self.frequency_res;
                (index, max, frequency)
            })
            .collect()
    }

    pub fn graph_ref(&self) -> &Vec<Vec<f32>> {
        &self.graph
    }
}

fn split_by_timestep<T: Clone>(vector: &Vec<T>, samples: usize) -> Vec<Vec<T>> {
    let mut iter = vector.chunks(samples);
    let mut split: Vec<Vec<T>> = Vec::new();

    let mut ptr = iter.next();
    while !ptr.is_none() {
        let val = ptr.unwrap_or(&[]);
        split.push(val.to_vec());
        ptr = iter.next();
    }

    split
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
    use super::*;

    #[test]
    fn testing_split() {
        let sample_vec: Vec<f32> = vec![3.0; 4];
        let answer_vec: Vec<Vec<f32>> = vec![vec![3.0, 3.0]; 2];
        assert_eq!(split_by_timestep(&sample_vec, 2), answer_vec);
    }

    #[test]
    fn uneven_split() {
        let sample_vec: Vec<f32> = vec![3.0; 7];
        let mut answer_vec: Vec<Vec<f32>> = vec![vec![3.0, 3.0]; 3];
        answer_vec.push(vec![3.0]);

        assert_eq!(split_by_timestep(&sample_vec, 2), answer_vec);
    }
}
