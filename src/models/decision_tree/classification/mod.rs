#![allow(unknown_lints)]

mod histogram;
pub mod impurity;
mod split_leaves;
mod streaming_classification_tree;

pub use self::streaming_classification_tree::StreamingClassificationTree;

use super::tree::*;
use self::impurity::*;
use fnv::FnvHashMap;
use data::TrainingData;
use std::fmt::Debug;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::*;
use timely::dataflow::{Scope, Stream};
use timely::progress::nested::product::Product;
use timely::progress::Timestamp;
use timely::Data;

/// Extension trait for a stream of `TrainingData<T, L>`
trait SegmentTrainingData<S: Scope, T: Data, L: Data> {
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
        self.unary_frontier(Pipeline, "SegmentTrainingData", |_| {
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

                stash.retain(|time, _| !input.frontier().less_equal(time.time()));
            }
        })
    }
}

/*trait Predict<S: Scope, In: Data, P: Data> {
    fn predict(inputs: Stream<S, In>) -> Stream<S, P>;
}*/
