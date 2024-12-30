// builtin 
use std::{ffi::OsStr, fs::File, path::Path};

// external
use symphonia::core::{
    audio::{AudioBufferRef, Signal},
    codecs::{DecoderOptions, CODEC_TYPE_NULL},
    errors::Error,
    formats::{FormatOptions, Track},
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::{Hint, ProbeResult},
};

// internal


pub struct AudioConfig<'a> {
    file: File,
    extension: &'a str,
}

impl<'a> AudioConfig<'a> {
    //don't get lifetime tbh
    pub fn new(file_path: &str) -> AudioConfig {
        let file: File = File::open(file_path).expect("Failed to open file!");
        let extension = Path::new(file_path)
            .extension()
            .and_then(OsStr::to_str)
            .expect("Invalid file type!");
        AudioConfig { file, extension }
    }
}

pub struct PCMBuffer {
    pub samples: Vec<f32>,
}

pub fn audio_to_pcm(cfg: AudioConfig) -> PCMBuffer {
    let mut samples: Vec<f32> = Vec::new();

    let file: File = cfg.file;

    let mss: MediaSourceStream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint: Hint = Hint::new();
    hint.with_extension(cfg.extension);

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed: ProbeResult = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .expect("Unsupported Format!");

    let mut format = probed.format;

    let track: &Track = format
        .tracks()
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

    PCMBuffer { samples }
}
