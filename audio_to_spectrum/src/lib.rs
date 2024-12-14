use std::fs::File;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let path: String = String::from("../Chopin_Op9_No2.mp3");
        let samples: Vec<f32> = mp3_to_pcm(&path);
        println!("recorded sample number: {}", samples.len());
    }
}
