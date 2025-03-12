// builtin
use std::{mem::take, ops::Range};

// external

// internal
pub use crate::constants::*;

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

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct NoteEvent {
    time_delta_ms: f32,
    note: Note,
}

impl NoteEvent {
    pub fn new(time_delta_ms: f32, key: u8, note_on: bool) -> NoteEvent {
        NoteEvent {
            time_delta_ms,
            note: Note::new(key, note_on),
        }
    }

    pub fn get_time_delta(&self) -> f32 {
        self.time_delta_ms
    }

    pub fn get_note_ref(&self) -> &Note {
        &self.note
    }

    pub fn get_note(self) -> Note {
        self.note
    }
}

//TODO: change this implementation to be event based rather than time based,
#[derive(Debug, Clone)]
pub struct MIDIEncoding {
    encoding: Vec<Chord>,
}

impl MIDIEncoding {
    pub fn new(encoding: Vec<Chord>) -> MIDIEncoding {
        MIDIEncoding { encoding }
    }

    pub fn from_vector(output_vecs: Vec<Vec<f32>>, cutoff: f32) -> MIDIEncoding {
        let mut chords: Vec<Chord> = Vec::new();

        for vec in output_vecs.iter() {
            let chord = Chord::from_vec(vec, cutoff);
            chords.push(chord);
        }

        MIDIEncoding::new(chords)
    }

    pub fn get_encoding(&self) -> &Vec<Chord> {
        &self.encoding
    }

    pub fn len(&self) -> usize {
        self.encoding.len()
    }

    pub fn print(&self) -> String {
        let mut result = String::new();
        for (i, chord) in self.encoding.iter().enumerate() {
            if !chord.is_none() {
                result.push_str(&format!("Event: {}, Chord: {:?}\n", i, chord));
            }
        }

        if result.is_empty() {
            return "Empty Encoding".to_string();
        }

        result
    }
}

pub struct EncodingData {
    events: Vec<NoteEvent>,
    has_limit: bool,
    time_limit_ms: usize,
}

impl EncodingData {
    pub fn new(events: Vec<NoteEvent>) -> EncodingData {
        EncodingData {
            events,
            has_limit: false,
            time_limit_ms: 0,
        }
    }

    pub fn with_limit(events: Vec<NoteEvent>, limit_ms: usize) -> EncodingData {
        EncodingData {
            events,
            has_limit: true,
            time_limit_ms: limit_ms,
        }
    }

    pub fn get_events(&mut self) -> Vec<NoteEvent> {
        take(&mut self.events)
    }

    pub fn continue_sampling(&self, timestep_ms: u32) -> bool {
        if self.has_limit {
            (timestep_ms as usize) <= self.time_limit_ms
        } else {
            true
        }
    }

    pub fn time_limit_ms(&self) -> usize {
        self.time_limit_ms
    }
}

pub struct EncodingIndex;

impl EncodingIndex {
    /*
     88 keys = [21, 108]
     indices 2x = on/off, 2x + 1 for note ID
    */
    pub fn key_index(key: u8) -> usize {
        (2 * (key - 21) + 1) as usize
    }

    pub fn key_of(index: usize) -> u8 {
        ((index - 1) / 2 + 21) as u8
    }

    pub fn on_off_index(key: u8) -> usize {
        (2 * (key - 21)) as usize
    }

    pub fn key_range() -> Range<usize> {
        1..KEY_RANGE
    }

    pub fn time_delta_index() -> usize {
        KEY_RANGE + TIME_DELTA_INDEX
    }

    pub fn start_index() -> usize {
        KEY_RANGE + START_INDEX
    }

    pub fn end_index() -> usize {
        KEY_RANGE + END_INDEX
    }
}

#[derive(Clone, Debug)]
pub struct Chord {
    notes: Vec<Note>,
    is_none: bool,
    time_delta: f32,
    chord_type: ChordType,
}

impl Chord {
    pub fn new(note: NoteEvent, time_delta: f32) -> Chord {
        let notes: Vec<Note> = vec![note.get_note()];

        Chord {
            notes,
            is_none: false,
            time_delta,
            chord_type: ChordType::Chord,
        }
    }

    pub fn none() -> Chord {
        Chord {
            notes: Vec::new(),
            is_none: true,
            time_delta: 0.0,
            chord_type: ChordType::None,
        }
    }

    pub fn start() -> Chord {
        Chord {
            notes: Vec::new(),
            is_none: true,
            time_delta: 0.0,
            chord_type: ChordType::Start,
        }
    }

    pub fn end() -> Chord {
        Chord {
            notes: Vec::new(),
            is_none: true,
            time_delta: 0.0,
            chord_type: ChordType::End,
        }
    }

    pub fn from_vec(v: &Vec<f32>, cutoff: f32) -> Chord {
        if v.len() != ENCODING_LENGTH {
            panic!("Not a valid note encoding");
        }

        let mut notes: Vec<Note> = Vec::new();
        let time_delta: f32 = v[EncodingIndex::time_delta_index()];

        for i in EncodingIndex::key_range() {
            if v[i].is_nan() {
                panic!("Cannot make chord from NaN!");
            }
            if i % 2 != 1 {
                continue;
            }
            if v[i] >= cutoff {
                let key: u8 = EncodingIndex::key_of(i);
                let note: Note = Note::new(key, v[i - 1] >= cutoff);
                notes.push(note);
            }
        }

        if notes.is_empty() {
            return Chord::none();
        }

        Chord {
            notes,
            is_none: false,
            time_delta,
            chord_type: ChordType::Chord,
        }
    }

    pub fn reset(&mut self) {
        self.is_none = true;
        self.notes.clear();
    }

    pub fn is_none(&self) -> bool {
        self.is_none
    }

    fn add_note(&mut self, note: Note, time_delta: f32) {
        if self.is_none {
            self.time_delta = time_delta;
            self.is_none = false;
            self.chord_type = ChordType::Chord;
        }

        self.notes.push(note);
    }

    pub fn try_add(&mut self, note: Note, time_delta: f32) -> AddNoteResult {
        let mut same_event: bool = false;

        for n in self.notes.iter() {
            if n.get_key() == note.get_key() && n.is_note_on() == note.is_note_on() {
                same_event = true;
            }
        }

        if same_event {
            return AddNoteResult::Duplicate;
        } else {
            self.add_note(note, time_delta);
            return AddNoteResult::Ok;
        }
    }

    pub fn get_events(&self) -> Vec<NoteEvent> {
        let mut events: Vec<NoteEvent> = Vec::new();

        for note in self.notes.iter() {
            let event: NoteEvent =
                NoteEvent::new(self.time_delta, note.get_key(), note.is_note_on());
            events.push(event);
        }

        events
    }

    pub fn get_encoding(&self) -> Vec<f32> {
        let mut note_encoding: Vec<f32> = vec![0.0; ENCODING_LENGTH];

        for note in self.notes.iter() {
            let key: u8 = note.get_key();
            note_encoding[EncodingIndex::key_index(key)] = 1.0;
            if note.is_note_on() {
                note_encoding[EncodingIndex::on_off_index(key)] = 1.0;
            }
        }

        note_encoding[EncodingIndex::time_delta_index()] = self.time_delta;

        note_encoding
    }

    pub fn get_notes(&mut self) -> Vec<Note> {
        take(&mut self.notes)
    }
}

#[derive(Clone, Debug)]
pub enum ChordType {
    Chord,
    Start,
    End,
    None,
}

pub enum AddNoteResult {
    Ok,
    Duplicate,
}
