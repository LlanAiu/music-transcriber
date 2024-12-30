// builtin
use std::mem::take;

// external

// internal
use crate::constants::ENCODING_LENGTH;



#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Note {
    key: u8,
    on: bool,
}

impl Note {
    pub fn new(key: u8, on: bool) -> Note {
        Note { key, on }
    }

    pub fn get_key(&self) -> u8 {
        self.key
    }

    pub fn is_note_on(&self) -> bool {
        self.on
    }
}

#[derive(Debug, PartialEq, PartialOrd, Eq, Ord, Clone)]
pub struct NoteEvent {
    timestamp_ms: u32,
    note: Note
}

impl NoteEvent {
    pub fn new(timestamp: u32, key: u8, note_on: bool) -> NoteEvent {
        NoteEvent {
            timestamp_ms: timestamp,
            note: Note::new(key, note_on)
        }
    }

    pub fn get_timestamp(&self) -> u32 {
        self.timestamp_ms
    }

    pub fn get_note_ref(&self) -> &Note {
        &self.note
    }

    pub fn get_note(self) -> Note {
        self.note
    }
}

#[derive(Debug)]
pub struct MIDIEncoding {
    timestep_ms: f32,
    encoding: Vec<Chord>,
}

impl MIDIEncoding {
    pub fn new(timestep_ms: f32, encoding: Vec<Chord>) -> MIDIEncoding {
        MIDIEncoding {timestep_ms, encoding}
    }

    pub fn get_timestep(&self) -> f32 {
        self.timestep_ms
    }

    pub fn get_encoding(&self) -> &Vec<Chord> {
        &self.encoding
    }

    pub fn len(&self) -> usize {
        self.encoding.len()
    }
}

pub struct EncodingData {
    events: Vec<NoteEvent>,
    timestep_ms: f32
}

impl EncodingData {
    pub fn new(events: Vec<NoteEvent>, timestep_ms: f32) -> EncodingData {
        EncodingData {events, timestep_ms}
    }

    pub fn get_events(self) -> Vec<NoteEvent> {
        self.events
    }

    pub fn get_timestep(&self) -> f32 {
        self.timestep_ms
    }
}


#[derive(Clone, Debug)]
pub struct Chord {
    notes: Vec<Note>,
    is_none: bool,
}

impl Chord {
    pub fn new(note: NoteEvent) -> Chord {
        let notes: Vec<Note> = vec![note.get_note()];

        Chord {
            notes,
            is_none: false,
        }
    }

    pub fn none() -> Chord {
        Chord {
            notes: Vec::new(),
            is_none: true
        }
    }

    pub fn from_vec(v: &Vec<f32>) -> Chord {
        if v.len() != ENCODING_LENGTH {
            panic!("Not a valid note encoding");
        }
    
        let mut notes: Vec<Note> = Vec::new();
    
        for i in 1..v.len() {
            if i % 2 != 1 {
                continue;
            }
            if v[i] == 1.0 {
                let key: u8 = Chord::key_of(i);
                let note: Note = Note::new(key, v[i - 1] == 1.0);
                notes.push(note);
            }
        }
    
        if notes.is_empty() {
            return Chord::none();
        }

        Chord {
            notes,
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

    pub fn reset(&mut self) {
        self.is_none = true;
        self.notes.clear();
    }

    pub fn is_none(&self) -> bool {
        self.is_none
    }

    fn add_note(&mut self, note: Note) {
        self.is_none = false;
        self.notes.push(note);
    }

    pub fn try_add(&mut self, note: Note) -> AddNoteResult {
        let mut same_key: bool = false;
        let mut same_event: bool = false;

        for n in self.notes.iter() {
            if n.get_key() == note.get_key() {
                same_key = true;

                if n.is_note_on() == note.is_note_on() {
                    same_event = true;
                }
            }
        }
        
        if same_event {
            return AddNoteResult::Duplicate;
        } else if same_key {
            return AddNoteResult::PushToNext(note);
        } else {
            self.add_note(note);
            return AddNoteResult::Ok;
        }
    }

    pub fn get_events(&self, timestamp_ms: u32) -> Vec<NoteEvent> {
        let mut events: Vec<NoteEvent> = Vec::new();

        for note in self.notes.iter() {
            let event: NoteEvent = NoteEvent::new(timestamp_ms, note.get_key(), note.is_note_on());
            events.push(event);
        }

        events
    }

    pub fn get_encoding(&self) -> Vec<f32> {
        let mut note_encoding: Vec<f32> = vec![0.0; ENCODING_LENGTH];
        
        for note in self.notes.iter() {
            let key: u8 = note.get_key();
            note_encoding[Chord::key_index(key)] = 1.0;
            if note.is_note_on() {
                note_encoding[Chord::on_off_index(key)] = 1.0;
            } 
        }

        note_encoding
    }

    pub fn get_notes(&mut self) -> Vec<Note> {
        take(&mut self.notes)
    }
}

pub enum AddNoteResult {
    Ok,
    Duplicate,
    PushToNext(Note)
}