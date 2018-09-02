use data::dataflow::{ApplyLatest, InitEachTime, Timer};
use data::serialization::*;
use data::TrainingData;
use models::decision_tree::histogram_generics::{ContinuousValue, DiscreteValue};
use models::decision_tree::operators::{AggregateHistograms, CollectHistograms, SplitLeaves};
use models::decision_tree::regression::histogram::loss_functions::TrimmedLadWeightedLoss;
use models::decision_tree::regression::histogram::TargetValueHistogramSet;
use models::decision_tree::tree::DecisionTree;
use models::decision_tree::tree::DecisionTreeError;
use models::*;
use std::marker::PhantomData;
use timely::dataflow::operators::*;
use timely::dataflow::{Scope, Stream};
use timely::ExchangeData;
use std::time::Duration;

#[derive(Clone, Abomonation, Debug)]
pub struct StreamingRegressionTree<T, L> {
    levels: u64,
    points_per_worker: u64,
    bins: usize,
    trim_ratio: L,
    _t: PhantomData<T>,
}

impl<T, L> StreamingRegressionTree<T, L> {
    /// Creates a new model instance
    pub fn new(levels: u64, points_per_worker: u64, bins: usize, trim_ratio: L) -> Self {
        StreamingRegressionTree {
            levels,
            points_per_worker,
            bins,
            trim_ratio,
            _t: PhantomData,
        }
    }
}

impl<T: ExchangeData, L: ExchangeData> ModelAttributes for StreamingRegressionTree<T, L> {
    type TrainingResult = DecisionTree<T, L>;
}

impl<T: ExchangeData, L: ExchangeData> LabelingModelAttributes for StreamingRegressionTree<T, L> {
    type Predictions = AbomonableArray1<L>;

    type PredictErr = DecisionTreeError;
}

impl<S: Scope, T: DiscreteValue, L: ContinuousValue> Train<S, StreamingRegressionTree<T, L>>
    for Stream<S, TrainingData<T, L>>
{
    fn train(&self, model: &StreamingRegressionTree<T, L>) -> Stream<S, DecisionTree<T, L>> {
        let levels = model.levels;

        let init_tree = vec![DecisionTree::<T, L>::default()].init_each_time(self);

        self.scope().scoped::<u64, _, _>(|tree_iter_scope| {
            let (loop_handle, cycle) = tree_iter_scope.loop_variable(model.levels, 1);
            let (trees, timer) = init_tree
                .enter(tree_iter_scope)
                .concat(&cycle)
                .inspect_time(|time, _| debug!("Begin decision tree iteration {}", time.inner))
                .collect_histograms::<TargetValueHistogramSet<T, L>>(
                    &self.enter(tree_iter_scope),
                    model.bins,
                    model.points_per_worker as usize,
                )
                .aggregate_histograms::<TargetValueHistogramSet<T, L>>()
                .split_leaves(model.levels, TrimmedLadWeightedLoss(model.trim_ratio))
                .inspect_time(|time, (split_leaves, tree)| {
                    debug!(
                        "Split {} leaf nodes in iteration {}",
                        split_leaves, time.inner
                    );
                    debug!("Updated tree: {:?}", tree);
                })
                .map(move |(_, tree)| tree)
                .timer();
            let (iterate, finished_tree) = trees.branch(move |time, _| time.inner >= levels);

            timer.inspect_time(|time, result| {
                let d: Duration = (*result).into();
                info!("{:?}: {:?}", time, d);
            });

            iterate.broadcast().connect_loop(loop_handle);
            finished_tree.leave()
        })
    }
}

impl<S, T, L> Predict<S, StreamingRegressionTree<T, L>, DecisionTreeError>
    for Stream<S, AbomonableArray2<T>>
where
    S: Scope,
    T: ExchangeData + DiscreteValue,
    L: ExchangeData + ContinuousValue,
{
    fn predict(
        &self,
        _model: &StreamingRegressionTree<T, L>,
        train_results: Stream<S, DecisionTree<T, L>>,
    ) -> Stream<S, Result<AbomonableArray1<L>, ModelError<DecisionTreeError>>> {
        train_results.apply_latest(self, |_time, tree, samples| {
            tree.predict_samples(&samples).map(Into::into)
        })
    }
}
