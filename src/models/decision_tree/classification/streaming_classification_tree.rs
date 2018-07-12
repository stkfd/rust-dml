use data::dataflow::{ApplyLatest};
use data::serialization::*;
use data::TrainingData;
use models::decision_tree::classification::histogram::FeatureValueHistogramSet;
use models::decision_tree::histogram_generics::*;
use models::decision_tree::operators::*;
use models::decision_tree::split_improvement::SplitImprovement;
use models::decision_tree::tree::DecisionTree;
use models::decision_tree::tree::DecisionTreeError;
use models::ModelError;
use models::{ModelAttributes, Predict, PredictSamples, Train};
use std::marker::PhantomData;
use timely::dataflow::operators::*;
use timely::dataflow::{Scope, Stream};
use timely::Data;
use timely::ExchangeData;

/// Supervised model that builds a classification tree from streaming data
#[derive(Abomonation, Clone)]
pub struct StreamingClassificationTree<I: SplitImprovement<T, L> + Data, T, L> {
    levels: u64,
    points_per_worker: u64,
    bins: usize,
    impurity_algo: I,
    _t: PhantomData<T>,
    _l: PhantomData<L>,
}

impl<I, T, L> StreamingClassificationTree<I, T, L>
where
    T: ExchangeData + ContinuousValue,
    L: ExchangeData + DiscreteValue,
    I: Data + SplitImprovement<T, L, HistogramData = FeatureValueHistogramSet<T, L>>,
{
    /// Creates a new model instance
    pub fn new(levels: u64, points_per_worker: u64, bins: usize, impurity_algo: I) -> Self {
        StreamingClassificationTree {
            levels,
            points_per_worker,
            bins,
            impurity_algo,
            _t: PhantomData,
            _l: PhantomData,
        }
    }
}

impl<I, T, L> ModelAttributes for StreamingClassificationTree<I, T, L>
where
    T: ExchangeData + ContinuousValue,
    L: ExchangeData + DiscreteValue,
    I: Data + SplitImprovement<T, L, HistogramData = FeatureValueHistogramSet<T, L>>,
{
    type LabeledSamples = TrainingData<T, L>;
    type UnlabeledSamples = AbomonableArray2<T>;
    type Predictions = AbomonableArray1<L>;
    type TrainingResult = DecisionTree<T, L>;
}

impl<S, I, T, L> Train<S, StreamingClassificationTree<I, T, L>> for Stream<S, TrainingData<T, L>>
where
    S: Scope,
    T: ExchangeData + ContinuousValue,
    L: ExchangeData + DiscreteValue,
    I: Data + SplitImprovement<T, L, HistogramData = FeatureValueHistogramSet<T, L>>,
{
    fn train(
        &self,
        model_attributes: &StreamingClassificationTree<I, T, L>,
    ) -> Stream<S, DecisionTree<T, L>> {
        let mut scope = self.scope();
        let levels = model_attributes.levels;

        scope.scoped::<u64, _, _>(|tree_iter_scope| {
            let init_tree = if tree_iter_scope.index() == 0 {
                vec![DecisionTree::<T, L>::default()]
            } else {
                vec![]
            };

            let (loop_handle, cycle) = tree_iter_scope.loop_variable(model_attributes.levels, 1);
            let (iterate, finished_tree) = init_tree
                .to_stream(tree_iter_scope)
                .concat(&cycle)
                .inspect(|x| info!("Begin tree iteration: {:?}", x))
                .broadcast()
                .create_histograms::<FeatureValueHistogramSet<T, L>>(
                    &self.enter(tree_iter_scope),
                    model_attributes.bins,
                    model_attributes.points_per_worker as usize,
                )
                .aggregate_histograms::<FeatureValueHistogramSet<T, L>>()
                .split_leaves(
                    model_attributes.levels,
                    model_attributes.impurity_algo.clone(),
                )
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

impl<S, I, T, L>
    Predict<S, StreamingClassificationTree<I, T, L>, DecisionTreeError>
    for Stream<S, AbomonableArray2<T>>
where
    S: Scope,
    T: ExchangeData + ContinuousValue,
    L: ExchangeData + DiscreteValue,
    I: Data + SplitImprovement<T, L, HistogramData = FeatureValueHistogramSet<T, L>>,
{
    fn predict(
        &self,
        _model: &StreamingClassificationTree<I, T, L>,
        train_results: Stream<S, DecisionTree<T, L>>,
    ) -> Stream<S, Result<AbomonableArray1<L>, ModelError<DecisionTreeError>>> {
        train_results.apply_latest(self, |_time, tree, samples| {
            tree.predict_samples(&samples)
        })
    }
}
