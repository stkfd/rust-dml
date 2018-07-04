use data::dataflow::{Branch, ExchangeEvenly, SegmentTrainingData};
use data::serialization::*;
use data::TrainingData;
use fnv::FnvHashMap;
use models::decision_tree::histogram_generics::{ContinuousValue, DiscreteValue};
use models::decision_tree::operators::{AggregateHistograms, CreateHistograms, SplitLeaves};
use models::decision_tree::regression::histogram::loss_functions::TrimmedLadWeightedLoss;
use models::decision_tree::regression::histogram::TargetValueHistogramSet;
use models::decision_tree::tree::DecisionTree;
use models::StreamingSupModel;
use std::fmt::Debug;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::*;
use timely::dataflow::{Scope, Stream};
use timely::ExchangeData;
use Result;

pub struct StreamingRegressionTree<L> {
    levels: u64,
    points_per_worker: u64,
    bins: usize,
    trim_ratio: L,
}

impl<L> StreamingRegressionTree<L> {
    /// Creates a new model instance
    pub fn new(levels: u64, points_per_worker: u64, bins: usize, trim_ratio: L) -> Self {
        StreamingRegressionTree {
            levels,
            points_per_worker,
            bins,
            trim_ratio,
        }
    }
}

impl<T, L>
    StreamingSupModel<
        TrainingData<T, L>,
        AbomonableArray2<T>,
        AbomonableArray1<L>,
        DecisionTree<T, L>,
    > for StreamingRegressionTree<L>
where
    T: DiscreteValue + ExchangeData,
    L: ContinuousValue + ExchangeData,
{
    /// Predict output from inputs. Waits until a decision tree has been received before processing
    /// any data coming in on the stream of samples. When a tree has been received, all sample data
    /// is immediately processed using the latest received decision tree.
    fn predict<S: Scope>(
        &mut self,
        training_results: Stream<S, DecisionTree<T, L>>,
        inputs: Stream<S, AbomonableArray2<T>>,
    ) -> Result<Stream<S, AbomonableArray1<L>>> {
        let predictions =
            training_results.binary(&inputs, Pipeline, Pipeline, "Predict", |_, _| {
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
                                .entry(time.retain())
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
        data: Stream<S, TrainingData<T, L>>,
    ) -> Result<Stream<S, DecisionTree<T, L>>> {
        let levels = self.levels;

        let results = scope.scoped::<u64, _, _>(|data_segments_scope| {
            let training_data = data
                .enter(data_segments_scope)
                .segment_training_data(data_segments_scope.peers() as u64 * self.points_per_worker)
                .exchange_evenly();

            data_segments_scope
                .scoped::<u64, _, _>(|tree_iter_scope| {
                    let init_tree = if tree_iter_scope.index() == 0 {
                        vec![DecisionTree::<T, L>::default()]
                    } else {
                        vec![]
                    };

                    let (loop_handle, cycle) = tree_iter_scope.loop_variable(self.levels, 1);
                    let histograms_and_trees = init_tree
                        .to_stream(tree_iter_scope)
                        .concat(&cycle)
                        .inspect(|x| info!("Begin tree iteration: {:?}", x))
                        .broadcast()
                        .create_histograms::<TargetValueHistogramSet<T, L>>(
                            &training_data.enter(tree_iter_scope),
                            self.bins,
                            self.points_per_worker as usize,
                        )
                        .aggregate_histograms();

                    let (iterate, finished_tree) = SplitLeaves::split_leaves(
                        &histograms_and_trees,
                        self.levels,
                        TrimmedLadWeightedLoss(self.trim_ratio),
                        self.bins as u64,
                    ).map(move |(split_leaves, tree)| {
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