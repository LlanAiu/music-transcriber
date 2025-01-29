// builtin

// external

// internal

use audio_to_spectrum::spectrograph::{constants, Spectrograph};
use midi_encoder::types::{MIDIEncoding, ENCODING_LENGTH};

use crate::{rnn::RNN, types::{activation::init_registry, ActivationConfig, ParameterConfig, WeightConfig}};


pub trait Translate {
    fn translate_spectrum(&mut self, spectrum: Spectrograph) -> MIDIEncoding;

    fn update(&mut self, spectrum: Spectrograph, encoding: MIDIEncoding);
}

pub struct RNNConverter {
    rnn: RNN,
    batch: usize
}

impl RNNConverter {
    pub fn new(
        layers: usize,
        units_by_layer: Vec<usize>,
        batch: usize,
        weights: WeightConfig, 
        mut activations: ActivationConfig
    ) -> RNNConverter {
        init_registry();

        let input_size: usize = (
            (constants::BINS_PER_OCTAVE as f32) * 
            (constants::MAX_FREQ / constants::MIN_FREQ).log2().ceil()
        ) as usize;

        let output_size: usize = ENCODING_LENGTH;

        let mut params: ParameterConfig = ParameterConfig::new(
            layers, 
            input_size, 
            output_size, 
            units_by_layer
        );
        
        let rnn: RNN = RNN::new(&mut params, weights, &mut activations);

        RNNConverter {
            rnn,
            batch
        }
    }

    pub fn from_file(path: &str, batch: usize) -> RNNConverter {
        let rnn: RNN = RNN::from_save(path);

        let input_size: usize = (
            (constants::BINS_PER_OCTAVE as f32) * 
            (constants::MAX_FREQ / constants::MIN_FREQ).log2().ceil()
        ) as usize;

        let output_size: usize = ENCODING_LENGTH;
        
        if rnn.input_dim() != input_size || rnn.output_dim() != output_size {
            panic!("Invalid configuration file for Audio-to-MIDI converter");
        }

        RNNConverter {
            rnn,
            batch
        }
    }
}

impl Translate for RNNConverter {
    fn translate_spectrum(&mut self, mut spectrum: Spectrograph) -> MIDIEncoding {
        let timestep_ms: f32 = spectrum.get_timestep();
        let freq_seq: Vec<Vec<f32>> = spectrum.graph();
        let output_seq: Vec<Vec<f32>> = self.rnn.predict(freq_seq);

        MIDIEncoding::from_vector(output_seq, timestep_ms)
    }
    
    fn update(&mut self, mut spectrum: Spectrograph, encoding: MIDIEncoding) {
        let seq: Vec<Vec<f32>> = spectrum.graph();
        let output_seq: Vec<Vec<f32>> = encoding.get_encoding().iter().map(|chord| {
            chord.get_encoding()
        }).collect();

        self.rnn.predict_and_update(seq, &output_seq, self.batch);
    }    
}
