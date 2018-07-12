#![feature(try_from, generic_associated_types)]

extern crate abomonation;
#[macro_use]
extern crate abomonation_derive;
extern crate csv;
#[macro_use]
extern crate failure;
extern crate itertools;
#[macro_use]
extern crate log;
extern crate serde;
#[cfg(test)]
#[macro_use]
extern crate serde_derive;
extern crate timely;
extern crate timely_communication;
#[macro_use(s)]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate num_traits;
extern crate rand;
#[cfg_attr(test, macro_use)]
extern crate approx;
extern crate fnv;
extern crate ordered_float;
extern crate vec_map;
#[macro_use]
extern crate derive_more;

pub mod data;
pub mod models;

#[cfg(test)]
mod tests;
