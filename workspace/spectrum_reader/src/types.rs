// builtin 
use std::mem::take;

// external

// internal
use audio_to_spectrum::spectrograph::Spectrograph;
use midi_encoder::types::MIDIEncoding;


pub trait Translator {
    fn translate_spectrum(&mut self, spectrum: Spectrograph, cutoff: f32) -> MIDIEncoding;

    fn update(&mut self, spectrum: Spectrograph, encoding: MIDIEncoding);
}

pub struct ConverterConfig {
    layers: usize, 
    units_by_layer: Vec<usize>,
    batch_size: usize
}

impl ConverterConfig {
    pub fn new(layers: usize, units_by_layer: Vec<usize>, batch_size: usize) -> ConverterConfig {
        ConverterConfig {
            layers,
            units_by_layer,
            batch_size,
        }
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn units_by_layer(&mut self) -> Vec<usize> {
        take(&mut self.units_by_layer)
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

