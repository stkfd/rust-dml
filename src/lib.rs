#![feature(try_from)]
#![allow(unknown_lints)]
#![cfg_attr(feature="profile", feature(plugin, custom_attribute))]
#![cfg_attr(feature="profile", plugin(flamer))]

extern crate abomonation;
#[macro_use]
extern crate abomonation_derive;
extern crate csv;
#[macro_use(s)]
extern crate ndarray;
#[macro_use]
extern crate failure;
extern crate itertools;
#[macro_use]
extern crate log;
extern crate fnv;
extern crate ndarray_linalg;
extern crate num_traits;
extern crate ordered_float;
extern crate rand;
extern crate serde;
extern crate timely;
extern crate timely_communication;
extern crate vec_map;
#[macro_use]
extern crate derive_more;
extern crate probability;

#[cfg_attr(test, macro_use)]
extern crate approx;

#[cfg(test)]
#[macro_use]
extern crate serde_derive;

#[cfg(feature="profile")]
extern crate flame;

pub mod data;
pub mod models;

#[cfg(test)]
mod tests;
