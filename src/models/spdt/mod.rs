#![allow(unknown_lints)]

mod histogram;
pub mod impurity;
mod split_leaves;
mod tree;

use self::histogram::operators::*;
use self::impurity::*;
use self::split_leaves::*;
use self::tree::*;
use data::dataflow::{Branch, ExchangeEvenly};
use data::serialization::*;
use fnv::FnvHashMap;
use models::StreamingSupModel;
use models::TrainingData;
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

impl<L, I: Impurity>
    StreamingSupModel<
        TrainingData<f64, L>,
        AbomonableArray2<f64>,
        AbomonableArray1<L>,
        DecisionTree<f64, L>,
    > for StreamingDecisionTree<I>
where
    L: Debug + ExchangeData + Into<usize> + Copy + Eq + Hash + 'static,
{
    /// Predict output from inputs. Waits until a decision tree has been received before processing
    /// any data coming in on the stream of samples. When a tree has been received, all sample data
    /// is immediately processed using the latest received decision tree.
    fn predict<S: Scope>(
        &mut self,
        training_results: Stream<S, DecisionTree<f64, L>>,
        inputs: Stream<S, AbomonableArray2<f64>>,
    ) -> Result<Stream<S, AbomonableArray1<L>>> {
        let predictions = training_results.binary(&inputs, Pipeline, Pipeline, "Predict", |_| {
            let mut current_tree = None;
            let mut input_stash = FnvHashMap::default();

            move |trees, inputs, output| {
                trees.for_each(|_, data| {
                    current_tree = Some(data.drain(..).last().expect("Latest decision tree"));
                });
                inputs.for_each(|time, data| {
                    if let Some(tree) = &current_tree {
                        for samples in data.drain(..) {
                            let samples = samples.view();
                            output
                                .session(&time)
                                .give(AbomonableArray1::from(tree.predict_samples(samples)));
                        }
                    } else {
                        input_stash
                            .entry(time)
                            .or_insert_with(Vec::new)
                            .extend(data.drain(..));
                    }
                });

                if let Some(tree) = &current_tree {
                    for (time, data) in &mut input_stash {
                        for samples in data.drain(..) {
                            let samples = samples.view();
                            output
                                .session(&time)
                                .give(AbomonableArray1::from(tree.predict_samples(samples)));
                        }
                    }
                }
                input_stash.retain(|_, data| !data.is_empty());
            }
        });
        Ok(predictions)
    }

    /// Train the model using inputs and targets.
    fn train<S: Scope>(
        &mut self,
        scope: &mut S,
        data: Stream<S, TrainingData<f64, L>>,
    ) -> Result<Stream<S, DecisionTree<f64, L>>> {
        let levels = self.levels;

        let results = scope.scoped::<u64, _, _>(|data_segments_scope| {
            let training_data = data.enter(data_segments_scope)
                .segment_training_data(data_segments_scope.peers() as u64 * self.points_per_worker)
                .exchange_evenly();

            data_segments_scope
                .scoped::<u64, _, _>(|tree_iter_scope| {
                    let init_tree = if tree_iter_scope.index() == 0 {
                        vec![DecisionTree::<f64, L>::default()]
                    } else {
                        vec![]
                    };

                    let (loop_handle, cycle) = tree_iter_scope.loop_variable(self.levels, 1);
                    let (iterate, finished_tree) = init_tree
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
                        .map(move |(split_leaves, tree)| {
                            info!("Split {} leaves", split_leaves);
                            tree
                        })
                        .branch(move |time, _| time.inner >= levels);
                    iterate.connect_loop(loop_handle);
                    finished_tree.leave()
                })
                .leave()
        });
        Ok(results)
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

/*trait Predict<S: Scope, In: Data, P: Data> {
    fn predict(inputs: Stream<S, In>) -> Stream<S, P>;
}*/
