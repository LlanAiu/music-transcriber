// builtin

// external

// internal


#[derive(Debug, PartialEq, PartialOrd, Eq, Ord)]
pub struct NoteEvent {
    timestamp_ms: u32,
    key: u8,
    note_on: bool,
}

impl NoteEvent {
    pub fn new(timestamp: u32, key: u8, note_on: bool) -> NoteEvent {
        NoteEvent {
            timestamp_ms: timestamp,
            key,
            note_on,
        }
    }

    pub fn get_timestamp(&self) -> u32 {
        self.timestamp_ms
    }

    pub fn get_key_index(&self) -> u8 {
        self.key
    }

    pub fn is_note_on(&self) -> bool {
        self.note_on
    }
}

#[derive(Debug)]
pub struct MIDIEncoding {
    timestep_ms: u32,
    encoding: Vec<Vec<f32>>,
}

impl MIDIEncoding {
    pub fn new(timestep_ms: u32, encoding: Vec<Vec<f32>>) -> MIDIEncoding {
        MIDIEncoding {timestep_ms, encoding}
    }

    pub fn get_timestep(&self) -> u32 {
        self.timestep_ms
    }

    pub fn get_encoding(&self) -> &Vec<Vec<f32>> {
        &self.encoding
    }
}

pub struct EncodingData<'a> {
    events: &'a Vec<NoteEvent>,
    timestep_ms: u32
}

impl<'a> EncodingData<'a> {
    pub fn new(events: &'a Vec<NoteEvent>, timestep_ms: u32) -> EncodingData {
        EncodingData {events, timestep_ms}
    }

    pub fn get_events(&self) -> &Vec<NoteEvent> {
        self.events
    }

    pub fn get_timestep(&self) -> u32 {
        self.timestep_ms
    }
}