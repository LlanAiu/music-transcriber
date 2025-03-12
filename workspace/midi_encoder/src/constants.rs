pub const ENCODING_LENGTH: usize = 88 * 2 + NON_KEY_VARS;
pub const KEY_RANGE: usize = 88 * 2;

/*
  1 - time_delta;
  2 - start;
  3 - end;
*/

pub const NON_KEY_VARS: usize = 3;

pub const TIME_DELTA_INDEX: usize = 0;
pub const START_INDEX: usize = 1;
pub const END_INDEX: usize = 2;
