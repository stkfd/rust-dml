use models::decision_tree::tree::DecisionTreeError;
use std::marker::PhantomData;
use data::dataflow::ApplyLatest;
use data::serialization::*;
use data::TrainingData;
use models::decision_tree::histogram_generics::{ContinuousValue, DiscreteValue};
use models::decision_tree::operators::{AggregateHistograms, CreateHistograms, SplitLeaves};
use models::decision_tree::regression::histogram::loss_functions::TrimmedLadWeightedLoss;
use models::decision_tree::regression::histogram::TargetValueHistogramSet;
use models::decision_tree::tree::DecisionTree;
use models::*;
use timely::dataflow::operators::*;
use timely::dataflow::{Scope, Stream};
use timely::ExchangeData;

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

impl<T: Data, L: Data> ModelAttributes for StreamingRegressionTree<T, L> {
    type LabeledSamples = TrainingData<T, L>;
    type UnlabeledSamples = AbomonableArray2<T>;
    type Predictions = AbomonableArray1<L>;
    type TrainingResult = DecisionTree<T, L>;
}

impl<S: Scope, T: DiscreteValue, L: ContinuousValue> Train<S, StreamingRegressionTree<T, L>>
    for Stream<S, TrainingData<T, L>>
{
    fn train(
        &self,
        model: &StreamingRegressionTree<T, L>,
    ) -> Stream<S, DecisionTree<T, L>> {
        let mut scope = self.scope();
        let levels = model.levels;

        scope.scoped::<u64, _, _>(|tree_iter_scope| {
            let init_tree = if tree_iter_scope.index() == 0 {
                vec![DecisionTree::<T, L>::default()]
            } else {
                vec![]
            };

            let (loop_handle, cycle) = tree_iter_scope.loop_variable(model.levels, 1);
            let (iterate, finished_tree) = init_tree
                .to_stream(tree_iter_scope)
                .concat(&cycle)
                .inspect(|x| info!("Begin tree iteration: {:?}", x))
                .broadcast()
                .create_histograms::<TargetValueHistogramSet<T, L>>(
                    &self.enter(tree_iter_scope),
                    model.bins,
                    model.points_per_worker as usize,
                )
                .aggregate_histograms::<TargetValueHistogramSet<T, L>>()
                .split_leaves(model.levels, TrimmedLadWeightedLoss(model.trim_ratio))
                .map(move |(split_leaves, tree)| {
                    info!("Split {} leaves", split_leaves);
                    tree
                })
                .branch(move |time, _| time.inner >= levels);

            iterate.connect_loop(loop_handle);
            finished_tree.leave()
        })
    }
}

impl<S, T, L>
    Predict<S, StreamingRegressionTree<T, L>, DecisionTreeError>
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
            tree.predict_samples(&samples)
        })
    }
}
