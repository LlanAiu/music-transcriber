// builtin 

// external

// internal
extern crate midi_encoder;
extern crate audio_to_spectrum;
use midi_encoder::types::MIDIEncoding;
use audio_to_spectrum::spectrograph::Spectrograph;

pub trait Transcribe {
    fn translate(&self, spectrum: Vec<Vec<f32>>) -> Vec<f32>;
}

pub trait Trainable {
    fn process(&mut self, input: Spectrograph, output: MIDIEncoding);

    fn record(&mut self, input: Vec<Vec<f32>>, output: Vec<f32>);
}