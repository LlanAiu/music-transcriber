// builtin

// external

// internal
mod pcm;
pub mod spectrograph;
use pcm::{audio_to_pcm, AudioConfig, PCMBuffer};
use spectrograph::{pcm_to_spectrograph, Spectrograph};


pub fn audio_to_spectrograph(file_path: &str) -> Spectrograph {
    let pcm: PCMBuffer = audio_to_pcm(AudioConfig::new(file_path));
    pcm_to_spectrograph(pcm)
}

pub fn get_sample_spectrograph(file_path: &str, len_sec: f32) -> Spectrograph {
    let samples: usize = (44100.0 * len_sec).floor() as usize;
    let pcm: PCMBuffer = audio_to_pcm(
        AudioConfig::with_limit(file_path, samples));
    pcm_to_spectrograph(pcm)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn pcm_test() {
        let path: String = String::from("./tests/700hz_test.mp3");
        let spectrograph: Spectrograph = audio_to_spectrograph(&path);
        println!("{}", spectrograph.graph_ref().len());
    }

    #[test]
    fn basic_spectrograph() {
        let path: String = String::from("./tests/700hz_test.mp3");
        let image: String = String::from("./tests/spectrographs/700hz_test_image.png");
        let spectrograph: Spectrograph = audio_to_spectrograph(&path);
        let _ = spectrograph.generate_heatmap(&image).expect("Failed to generate heatmap");
    }

    #[test]
    fn simple_melody() {
        let path: String = String::from("./tests/Happy_bday.mp3");
        let image: String = String::from("./tests/spectrographs/Happy_bday_image.png");
        let spectrograph: Spectrograph = audio_to_spectrograph(&path);
        println!("Vector length: {}", spectrograph.vector_dim());
        let _ = spectrograph.generate_heatmap(&image).expect("Failed to generate heatmap");
    }

    #[test]
    fn complex_melody() {
        let path: String = String::from("./tests/K331_15s.mp3");
        let image: String = String::from("./tests/spectrographs/K331_15s_image.png");
        let spectrograph: Spectrograph = audio_to_spectrograph(&path);
        println!("Vector length: {}", spectrograph.vector_dim());
        let _ = spectrograph.generate_heatmap(&image).expect("Failed to generate heatmap");
    }

    #[test]
    fn data_test() {
        let spectrograph: Spectrograph = audio_to_spectrograph("./tests/Data_test.mp3");
        println!("Vectors: {}", spectrograph.num_timestamps());
    }

    #[test]
    fn sample_data_test() {
        let spectrograph: Spectrograph = get_sample_spectrograph("./tests/Data_test.mp3", 3.0);
        println!("Vectors: {}", spectrograph.num_timestamps());
    }
}