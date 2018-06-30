//! Extension traits and other tools to deal with dataflow Streams.

use data::TrainingData;
use fnv::FnvHashMap;
use std::sync::mpsc;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::capture::Event;
use timely::dataflow::operators::*;
use timely::dataflow::{Scope, Stream};
use timely::progress::nested::product::Product;
use timely::progress::Timestamp;
use timely::Data;
use Result;

mod branch;
mod exchange_evenly;
mod random;

pub use self::branch::{Branch, BranchWhen};
pub use self::exchange_evenly::ExchangeEvenly;

/// A container for result data coming in asynchronously from somewhere. Internally uses the `std::sync::mpsc`
/// channels for receiving the data.
pub enum AsyncResult<T> {
    Receiver(mpsc::Receiver<T>),
    Data(T),
    Uninitialized,
}

impl<T> AsyncResult<T> {
    /// Try to retrieve the data contained within this Result.
    /// If the container currently contains a `Receiver`, this method blocks the thread while waiting
    /// for the data from the channel.
    /// Does nothing if the data was already retrieved, returns an error if the Result is uninitialized
    /// and does not yet contain a `Receiver`.
    pub fn fetch(&mut self) -> Result<()> {
        *self = match *self {
            AsyncResult::Receiver(ref receiver) => {
                let data = receiver.recv()?;
                AsyncResult::Data(data)
            }
            AsyncResult::Data(_) => return Ok(()),
            AsyncResult::Uninitialized => {
                return Err(format_err!(
                    "Attempted to retrieve training results, but the model was not trained"
                ))
            }
        };
        Ok(())
    }

    /// Tries to get the data using `fetch` and returns a reference to it if successful
    pub fn get(&mut self) -> Result<&T> {
        self.fetch()?;
        Ok(self.get_unchecked())
    }

    /// Returns a reference to the contained data, assuming that it was already retrieved. Panics if that is not the case.
    pub fn get_unchecked(&self) -> &T {
        match *self {
            AsyncResult::Data(ref data) => data,
            _ => panic!("Result did not contain data when get_unchecked was called"),
        }
    }

    /// Tries to get the data using `fetch` and returns it if successful
    pub fn take(mut self) -> Result<T> {
        self.fetch()?;
        Ok(self.take_unchecked())
    }

    /// Returns the contained data, assuming that it was already retrieved. Panics if that is not the case.
    pub fn take_unchecked(self) -> T {
        match self {
            AsyncResult::Data(data) => data,
            _ => panic!("Result did not contain data when take_unchecked was called"),
        }
    }
}

/// Supports extracting a sequence of timestamp and data. Unlike `Extract`, it doesn't
/// try to order the extracted data by its content, so it can extract any Data, regardless
/// of whether it implements `Ord`. Otherwise, it is identical to the `Extract` trait
/// from Timely Dataflow.
pub trait ExtractUnordered<T: Ord, D> {
    /// Converts `self` into a sequence of timestamped data.
    ///
    /// Currently this is only implemented for `Receiver<Event<T, D>>`, and is used only
    /// to easily pull data out of a timely dataflow computation once it has completed.
    fn extract_unordered(&self) -> Vec<(T, Vec<D>)>;
}

impl<T: Ord, D> ExtractUnordered<T, D> for ::std::sync::mpsc::Receiver<Event<T, D>> {
    fn extract_unordered(&self) -> Vec<(T, Vec<D>)> {
        let mut result = Vec::new();
        for event in self {
            if let Event::Messages(time, data) = event {
                result.push((time, data));
            }
        }
        result.sort_by(|x, y| x.0.cmp(&y.0));

        let mut current = 0;
        for i in 1..result.len() {
            if result[current].0 == result[i].0 {
                let dataz = ::std::mem::replace(&mut result[i].1, Vec::new());
                result[current].1.extend(dataz);
            } else {
                current = i;
            }
        }

        result.retain(|x| !x.1.is_empty());
        result
    }
}

/// Extension trait for a stream of `TrainingData<T, L>`
pub trait SegmentTrainingData<S: Scope, T: Data, L: Data> {
    /// Segments training data that comes in on one timestamp into a maximum of
    /// `items_per_segment` items. This is used so that on a continuous stream of
    /// data, trees can be created with reasonably sized chunks of data.
    /// Expects the scope where the operator is used to have a Product(Timestamp, u64)
    /// timestamp, where the inner timestamp will be used to indicate the segment number
    /// of data
    fn segment_training_data(&self, items_per_segment: u64) -> Stream<S, TrainingData<T, L>>;
}

impl<S: Scope<Timestamp = Product<Ts, u64>>, Ts: Timestamp, T: Data, L: Data>
    SegmentTrainingData<S, T, L> for Stream<S, TrainingData<T, L>>
{
    fn segment_training_data(&self, items_per_segment: u64) -> Stream<S, TrainingData<T, L>> {
        self.unary_frontier(Pipeline, "SegmentTrainingData", |_, _| {
            let mut stash: FnvHashMap<_, (Product<_, u64>, u64)> = FnvHashMap::default();

            move |input, output| {
                input.for_each(|cap, data| {
                    let time = stash
                        .entry(cap.clone())
                        .or_insert_with(|| (cap.time().clone(), 0));
                    for training_data in data.iter() {
                        time.1 += training_data.x().rows() as u64;
                        if time.1 > items_per_segment {
                            time.0.inner += 1;
                        }
                    }
                    output.session(&cap.delayed(&time.0)).give_content(data);
                });

                stash.retain(|time, _| !input.frontier().less_equal(time));
            }
        })
    }
}
