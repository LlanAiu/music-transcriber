mod pcm;
mod spectrograph;
use pcm::{audio_to_pcm, AudioConfig, PCMBuffer};
use spectrograph::Spectrograph;
use spectrograph::PCMConverter;

const FFT_LEN: usize = 16392;
const NUM_SAMPLES: usize = 2205;

pub fn audio_to_spectrograph(file_path: &str) -> Spectrograph {
    let pcm: PCMBuffer = audio_to_pcm(AudioConfig::new(file_path));
    let converter: PCMConverter = PCMConverter::new(FFT_LEN, NUM_SAMPLES);

    converter.to_spectrograph(pcm)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn pcm_test() {
        let path: String = String::from("../tests/Chopin_Op9_No2.mp3");
        let spectrograph: Spectrograph = audio_to_spectrograph(&path);
        println!("{}", spectrograph.graph_ref().len());
    }

    #[test]
    fn basic_spectrograph() {
        let path: String = String::from("../tests/700hz_test.mp3");
        let image: String = String::from("../tests/700hz_test_image.png");
        let spectrograph: Spectrograph = audio_to_spectrograph(&path);
        let _ = spectrograph.generate_heatmap(&image).expect("Failed to generate heatmap");
    }

    #[test]
    fn simple_melody() {
        let path: String = String::from("../tests/Happy_bday.mp3");
        let image: String = String::from("../tests/Happy_bday_spectrograph.png");
        let spectrograph: Spectrograph = audio_to_spectrograph(&path);
        let _ = spectrograph.generate_heatmap(&image).expect("Failed to generate heatmap");
    }

    #[test]
    fn complex_melody() {
        let path: String = String::from("../tests/K331_15s.mp3");
        let image: String = String::from("../tests/K331_15s_image.png");
        let spectrograph: Spectrograph = audio_to_spectrograph(&path);
        let _ = spectrograph.generate_heatmap(&image).expect("Failed to generate heatmap");
    }
}