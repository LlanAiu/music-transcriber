// builtin

// external

// internal
use crate::constants::ENCODING_LENGTH;


#[derive(Debug, PartialEq, PartialOrd, Eq, Ord, Clone)]
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
    timestep_ms: f32,
    encoding: Vec<Note>,
}

impl MIDIEncoding {
    pub fn new(timestep_ms: f32, encoding: Vec<Note>) -> MIDIEncoding {
        MIDIEncoding {timestep_ms, encoding}
    }

    pub fn get_timestep(&self) -> f32 {
        self.timestep_ms
    }

    pub fn get_encoding(&self) -> &Vec<Note> {
        &self.encoding
    }
}

pub struct EncodingData<'a> {
    events: &'a Vec<NoteEvent>,
    timestep_ms: f32
}

impl<'a> EncodingData<'a> {
    pub fn new(events: &'a Vec<NoteEvent>, timestep_ms: f32) -> EncodingData {
        EncodingData {events, timestep_ms}
    }

    pub fn get_events(&self) -> &Vec<NoteEvent> {
        self.events
    }

    pub fn get_timestep(&self) -> f32 {
        self.timestep_ms
    }
}

#[derive(Clone, Debug)]
pub struct Note {
    events: Vec<NoteEvent>,
    encoding: Vec<f32>,
    is_none: bool,
}

impl Note {
    pub fn new(note: NoteEvent) -> Note {
        let events: Vec<NoteEvent> = vec![note];
        Note {
            events,
            encoding: vec![0.0],
            is_none: false,
        }
    }

    pub fn none() -> Note {
        Note {
            events: Vec::new(),
            encoding: vec![0.0; ENCODING_LENGTH],
            is_none: true
        }
    }

    pub fn from_vec(v: Vec<f32>, timestamp_ms: u32) -> Note {
        if v.len() != ENCODING_LENGTH {
            panic!("Not a valid note encoding");
        }
    
        let mut events: Vec<NoteEvent> = Vec::new();
    
        for i in 1..v.len() {
            if i % 2 != 1 {
                continue;
            }
            if v[i] == 1.0 {
                let key: u8 = Note::key_of(i);
                let event: NoteEvent = NoteEvent::new(timestamp_ms, key, v[i - 1] == 1.0);
                events.push(event);
            }
        }
    
        if events.is_empty() {
            return Note::none();
        }

        Note {
            events,
            encoding: v,
            is_none: false
        }
    }

    /*
     88 keys = [21, 108]
     indices 2x = on/off, 2x + 1 for note ID
    */
    fn key_index(key: u8) -> usize {
        (2 * (key - 21) + 1) as usize
    }

    fn key_of(index: usize) -> u8 {
        ((index - 1) / 2 + 21) as u8
    }

    fn on_off_index(key: u8) -> usize {
        (2 * (key - 21)) as usize
    }

    fn get_encoding(note: &NoteEvent) -> Vec<f32> {
        let mut note_encoding: Vec<f32> = vec![0.0; ENCODING_LENGTH];
        let key: u8 = note.get_key_index();
        note_encoding[Note::key_index(key)] = 1.0;
        if note.is_note_on() {
            note_encoding[Note::on_off_index(key)] = 1.0;
        }

        note_encoding
    }

    pub fn reset(&mut self) {
        self.is_none = true;
        self.encoding = vec![0.0; ENCODING_LENGTH];
        self.events.clear();
    }

    pub fn is_none(&self) -> bool {
        self.is_none
    }

    fn add_note(&mut self, note: NoteEvent) {
        self.is_none = false;
        let encoding: Vec<f32> = Note::get_encoding(&note);
        let mut new_encoding: Vec<f32> = Vec::new();

        for i in 0..encoding.len() {
            let val = encoding[i] + self.encoding[i];
            new_encoding.push(val);
        }

        self.encoding = new_encoding;
        
        self.events.push(note);
    }

    pub fn try_add<'a>(&mut self, note: &'a NoteEvent) -> AddNoteResult<'a> {
        let key: u8 = note.get_key_index();
        let same_key: bool = self.encoding[Note::key_index(key)] == 1.0;
        let same_event: bool = self.events.contains(&note);
        
        if same_event {
            return AddNoteResult::Duplicate;
        } else if same_key {
            return AddNoteResult::PushToNext(note);
        } else {
            self.add_note(note.clone());
            return AddNoteResult::Ok;
        }
    }

    pub fn get_events(&self) -> Vec<NoteEvent> {
        self.events.clone()
    }

}

pub enum AddNoteResult<'a> {
    Ok,
    Duplicate,
    PushToNext(&'a NoteEvent)
}