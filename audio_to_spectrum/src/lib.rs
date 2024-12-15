use std::fs::File;

use rustfft::{num_complex::Complex, FftPlanner};
use symphonia::core::{audio::{AudioBufferRef, Signal}, codecs::{DecoderOptions, CODEC_TYPE_NULL}, errors::Error, formats::{FormatOptions, Track}, io::MediaSourceStream, meta::MetadataOptions, probe::{Hint, ProbeResult}};

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

pub fn mp3_to_pcm(file_path: &str) -> Vec<f32>{
    let mut samples: Vec<f32> = Vec::new();

    let file: File = File::open(file_path).expect("Failed to open file!");
    
    let mss: MediaSourceStream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint: Hint = Hint::new();
    hint.with_extension("mp3");

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed: ProbeResult = symphonia::default::get_probe()
        .format( 
            &hint,
            mss,
            &fmt_opts, 
            &meta_opts)
        .expect("Unsupported Format!");
        
    let mut format = probed.format;

    let track: &Track = format.tracks()
                            .iter()
                            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
                            .expect("No supported audio tracks found");
    
    let dec_opts: DecoderOptions = Default::default();

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .expect("Unsupported codec");

    let track_id = track.id;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(_)) => {
                // Handle end of stream error and exit the loop
                break;
            }
            Err(err) => {
                panic!("Failed reading packets with error: {}", err);
            }
        };

        while !format.metadata().is_latest() {
            format.metadata().pop();
        }

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(_decoded) => {
                match _decoded {
                    AudioBufferRef::F32(buf) => {
                        for &sample in buf.chan(0) {
                            samples.push(sample);
                        }
                    }
                    _ => {
                        // Handle other buffer types if necessary
                    }
                }
            }
            Err(Error::IoError(_)) => {
                continue;
            }
            Err(Error::DecodeError(_)) => {
                continue;
            }
            Err(err) => {
                panic!("{}", err);
            }
        }
    }

    samples
}

pub fn pcm_to_spectrograph(pcm: Vec<f32>) -> Vec<Vec<f32>> {
    let complex_data: Vec<Complex<f32>> = pcm.into_iter()
        .map(|sample| Complex { re: sample, im: 0.0})
        .collect();

    let sample_matrix: Vec<Vec<Complex<f32>>> = split_by_timestep(&complex_data , 735);
    let mut spectrograph: Vec<Vec<f32>> = Vec::new();

    let mut planner: FftPlanner<f32> = FftPlanner::new();
    let fft = planner.plan_fft_forward(2000);

    for mut sample  in sample_matrix {
        if sample.len() < 2000 {
            sample.resize(2000, Complex {re: 0.0, im: 0.0});
        }

        fft.process(&mut sample);

        let sample_graph: Vec<f32> = sample.iter()
            .map(|val| val.norm())
            .collect();

        spectrograph.push(sample_graph);
    }

    spectrograph
}

pub fn split_by_timestep<T: Clone> (vector: &Vec<T>, samples: usize) -> Vec<Vec<T>> {
    let mut iter = vector.chunks(samples);
    let mut split: Vec<Vec<T>> = Vec::new();
    
    let mut ptr = iter.next();
    while !ptr.is_none() {
        let val= ptr.unwrap_or(&[]);
        split.push(val.to_vec());
        ptr = iter.next();
    }

    split
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn it_works() {
        let path: String = String::from("../Chopin_Op9_No2.mp3");
        let samples: Vec<f32> = mp3_to_pcm(&path);
        println!("recorded sample number: {}", samples.len());
    }

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

    #[test]
    fn basic_spectrograph() {
        
    }
}