#![allow(unknown_lints)]

mod histogram;
pub mod impurity;
mod tree;
mod split_leaves;

use models::TrainingData;
use self::histogram::operators::*;
use self::impurity::*;
use self::tree::*;
use self::split_leaves::*;
use data::dataflow::ExchangeEvenly;
use data::serialization::*;
use fnv::FnvHashMap;
use models::StreamingSupModel;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::*;
use timely::dataflow::{Scope, Stream};
use timely::progress::nested::product::Product;
use timely::progress::Timestamp;
use timely::{Data, ExchangeData};
use Result;

/// Supervised models that builds a decision tree from streaming data
pub struct StreamingDecisionTree<I: Impurity> {
    levels: u64,
    points_per_worker: u64,
    bins: usize,
    impurity_algo: PhantomData<I>,
}

impl<I: Impurity> StreamingDecisionTree<I> {
    /// Creates a new model instance
    pub fn new(levels: u64, points_per_worker: u64, bins: usize) -> Self {
        StreamingDecisionTree {
            levels,
            points_per_worker,
            bins,
            impurity_algo: PhantomData,
        }
    }
}

impl<L, I: Impurity> StreamingSupModel<TrainingData<f64, L>, AbomonableArray2<f64>, (), ()>
    for StreamingDecisionTree<I>
where
    L: Debug + ExchangeData + Into<usize> + Copy + Eq + Hash + 'static,
{
    /// Predict output from inputs.
    fn predict<S: Scope>(
        &mut self,
        _scope: &mut S,
        _inputs: Stream<S, AbomonableArray2<f64>>,
    ) -> Result<Stream<S, ()>> {
        unimplemented!()
    }

    /// Train the model using inputs and targets.
    fn train<S: Scope>(
        &mut self,
        scope: &mut S,
        data: Stream<S, TrainingData<f64, L>>,
    ) -> Result<Stream<S, ()>> {
        scope.scoped::<u64, _, _>(|data_segments_scope| {
            let training_data = data.enter(data_segments_scope)
                .segment_training_data(data_segments_scope.peers() as u64 * self.points_per_worker)
                .exchange_evenly();

            data_segments_scope.scoped::<u64, _, _>(|tree_iter_scope| {
                let init_tree = if tree_iter_scope.index() == 0 {
                    vec![DecisionTree::<f64, L>::default()]
                } else {
                    vec![]
                };

                let (loop_handle, cycle) = tree_iter_scope.loop_variable(self.levels + 1, 1);
                init_tree
                    .to_stream(tree_iter_scope)
                    .concat(&cycle)
                    .inspect(|x| info!("Begin tree iteration: {:?}", x))
                    .broadcast()
                    .create_histograms(
                        &training_data.enter(tree_iter_scope),
                        self.bins,
                        self.points_per_worker as usize,
                    )
                    .aggregate_histograms()
                    .split_leaves::<I>(self.levels, self.bins as u64)
                    .map(|(split_leaves, tree)| {
                        info!("Split {} leaves", split_leaves);
                        tree
                    })
                    .connect_loop(loop_handle);
            });
        });
        Ok(vec![()].to_stream(scope))
    }
}

trait SegmentTrainingData<S: Scope, T: Data, L: Data> {
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
