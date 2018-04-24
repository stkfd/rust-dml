#![feature(try_from, associated_type_defaults, specialization)]

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

use std::result::Result as StdResult;

pub mod data;
pub mod models;

type Result<T> = StdResult<T, ::failure::Error>;

#[cfg(test)]
mod tests;
