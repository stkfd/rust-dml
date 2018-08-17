use data::dataflow::ApplyLatest;
use data::serialization::*;
use data::TrainingData;
use failure::Fail;
use fnv::FnvHashMap;
use models::*;
use models::decision_tree::histogram_generics::{BaseHistogram, ContinuousValue, Median};
use models::decision_tree::regression::histogram::Histogram;
use ndarray::{ScalarOperand, Zip};
use ndarray::prelude::*;
use num_traits::{Float, FromPrimitive, NumAssign, ToPrimitive, Zero};
use std::collections::hash_map::Entry::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use timely::{Data, ExchangeData};
use timely::dataflow::{operators::*, Scope, Stream};
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::generic::builder_rc::OperatorBuilder;
use timely::dataflow::scopes::Child;

#[derive(Clone, Abomonation)]
pub struct GradientBoostingRegression<InnerModel, T, L> {
    iterations: u64,
    inner_model: InnerModel,
    _t: PhantomData<T>,
    _l: PhantomData<L>,
}

impl<InnerModel, T, L> GradientBoostingRegression<InnerModel, T, L> {
    pub fn new(iterations: u64, inner_model: InnerModel) -> Self {
        GradientBoostingRegression {
            iterations,
            inner_model,
            _t: PhantomData,
            _l: PhantomData,
        }
    }
}

impl<InnerModel: SupModelAttributes, T: ExchangeData, L: ExchangeData> ModelAttributes
for GradientBoostingRegression<InnerModel, T, L>
{
    type UnlabeledSamples = AbomonableArray2<T>;
    type TrainingResult = BoostChain<InnerModel, T, L>;
}

impl<InnerModel: SupModelAttributes, T: ExchangeData, L: ExchangeData> SupModelAttributes
for GradientBoostingRegression<InnerModel, T, L> {
    type LabeledSamples = TrainingData<T, L>;
    type Predictions = AbomonableArray1<L>;
    type PredictErr = InnerModel::PredictErr;
}

#[derive(Clone, Abomonation, Debug)]
pub struct BoostChain<InnerModel: SupModelAttributes, T, L>(
    pub Vec<(L, InnerModel::TrainingResult)>,
    PhantomData<(InnerModel, T, L)>,
);

impl<InnerModel: SupModelAttributes, T, L> BoostChain<InnerModel, T, L> {
    pub fn new(items: Vec<(L, InnerModel::TrainingResult)>) -> BoostChain<InnerModel, T, L> {
        BoostChain(items, PhantomData)
    }

    pub fn push_item(&mut self, multiplier: L, item: InnerModel::TrainingResult) {
        self.0.push((multiplier, item))
    }
}

impl<InnerModel: SupModelAttributes, T, L> Default for BoostChain<InnerModel, T, L> {
    fn default() -> Self {
        Self::new(vec![])
    }
}

impl<A, InnerModel, T, L> PredictSamples<A, AbomonableArray1<L>, InnerModel::PredictErr>
for BoostChain<InnerModel, T, L>
    where
            for<'a> &'a A: AsArray<'a, T, Ix2>,
            L: Float + ScalarOperand,
            InnerModel: SupModelAttributes,
            for<'a> InnerModel::TrainingResult:
            PredictSamples<ArrayView2<'a, T>, AbomonableArray1<L>, InnerModel::PredictErr>,
{
    fn predict_samples(
        &self,
        a: &A,
    ) -> Result<AbomonableArray1<L>, ModelError<InnerModel::PredictErr>> {
        let view = a.into();
        let mut agg = Array1::zeros(view.rows());

        for (multi, training_output) in &self.0 {
            let prediction: Array1<L> = training_output.predict_samples(&view)?.into();
            agg = agg + prediction * *multi;
        }
        Ok(agg.into())
    }
}

impl<S, T, L, InnerModel, E> Predict<S, GradientBoostingRegression<InnerModel, T, L>, E>
for Stream<S, AbomonableArray2<T>>
    where
        S: Scope,
        T: ExchangeData,
        L: ExchangeData + Zero,
        E: Data + Fail,
        InnerModel: SupModelAttributes,
        BoostChain<InnerModel, T, L>: PredictSamples<AbomonableArray2<T>, AbomonableArray1<L>, E>,
{
    fn predict(
        &self,
        _model: &GradientBoostingRegression<InnerModel, T, L>,
        train_results: Stream<S, BoostChain<InnerModel, T, L>>,
    ) -> Stream<S, Result<AbomonableArray1<L>, ModelError<E>>> {
        train_results.apply_latest(self, |_time, boost_chain, samples| {
            boost_chain.predict_samples(&samples).map(Into::into)
        })
    }
}

impl<S, InnerModel, T, L> TrainMeta<S, GradientBoostingRegression<InnerModel, T, L>>
for Stream<S, TrainingData<T, L>>
    where
        S: Scope,
        T: Debug + ExchangeData,
        L: ContinuousValue + ScalarOperand + NumAssign + ToPrimitive + FromPrimitive,
        InnerModel: SupModelAttributes<Predictions=AbomonableArray1<L>>,
        InnerModel::TrainingResult: Debug,
        for<'b> InnerModel::TrainingResult:
        ExchangeData
        + PredictSamples<ArrayView2<'b, T>, AbomonableArray1<L>, InnerModel::PredictErr>,
        for<'a> Stream<Child<'a, S, u64>, TrainingData<T, L>>: Train<Child<'a, S, u64>, InnerModel>,
{
    fn train_meta(
        &self,
        model: &GradientBoostingRegression<InnerModel, T, L>,
    ) -> Stream<S, BoostChain<InnerModel, T, L>> {
        let mut scope = self.scope();
        let worker = scope.index();
        scope.scoped::<u64, _, _>(|boost_iter_scope| {
            let (loop_handle, cycle) = boost_iter_scope.loop_variable(model.iterations - 1, 1);
            let data = self
                .enter(boost_iter_scope)
                .concat(&cycle)
                .inspect_time(move |time, _| {
                    debug!("W{}: Received residuals (round {})", worker, time.inner)
                });

            let training_results =
                <Stream<_, _> as Train<_, InnerModel>>::train(&data, &model.inner_model)
                    .inspect_time(move |time, _| {
                        debug!(
                            "W{}: Completed training model to residuals (round {})",
                            worker, time.inner
                        )
                    });

            let (residuals_stream, boost_chain_stream) =
                data.calculate_residuals(model.clone(), &training_results);

            residuals_stream.connect_loop(loop_handle);
            boost_chain_stream
        })
    }
}

#[allow(type_complexity)]
pub trait CalculateResiduals<
    'a,
    S: Scope,
    Model,
    TrainingData: Data,
    InnerModel: SupModelAttributes,
    T,
    L,
>
{
    fn calculate_residuals(
        &self,
        model: Model,
        training_results: &Stream<Child<'a, S, u64>, InnerModel::TrainingResult>,
    ) -> (
        Stream<Child<'a, S, u64>, TrainingData>,
        Stream<S, BoostChain<InnerModel, T, L>>,
    );
}

#[allow(type_complexity)]
impl<'a, S, T, L, InnerModel>
CalculateResiduals<
    'a,
    S,
    GradientBoostingRegression<InnerModel, T, L>,
    TrainingData<T, L>,
    InnerModel,
    T,
    L,
> for Stream<Child<'a, S, u64>, TrainingData<T, L>>
    where
        T: ExchangeData + Debug,
        L: ExchangeData
        + Debug
        + ContinuousValue
        + ScalarOperand
        + NumAssign
        + ToPrimitive
        + FromPrimitive,
        S: Scope,
        InnerModel: SupModelAttributes<Predictions=AbomonableArray1<L>>,
        for<'b> InnerModel::TrainingResult:
        ExchangeData
        + Debug
        + PredictSamples<ArrayView2<'b, T>, AbomonableArray1<L>, InnerModel::PredictErr>,
{
    fn calculate_residuals(
        &self,
        model: GradientBoostingRegression<InnerModel, T, L>,
        training_results: &Stream<Child<'a, S, u64>, InnerModel::TrainingResult>,
    ) -> (
        Stream<Child<'a, S, u64>, TrainingData<T, L>>,
        Stream<S, BoostChain<InnerModel, T, L>>,
    ) {
        let worker = self.scope().index();
        let mut builder = OperatorBuilder::new("CalculateResiduals".to_owned(), self.scope());

        let mut training_data_input = builder.new_input(self, Pipeline);
        let mut training_result_input = builder.new_input(&training_results.broadcast(), Pipeline);

        let (mut residuals_output, residuals_stream) = builder.new_output();
        let (mut boost_chain_output, boost_chain_stream) = builder.new_output();

        builder.build(|_fsdd| {
            let mut training_data_stash = FnvHashMap::default();
            let mut training_result_stash = FnvHashMap::default();
            let mut boost_chain_stash = FnvHashMap::default();

            move |frontiers| {
                let mut residuals_handle = residuals_output.activate();
                let mut boost_chain_handle = boost_chain_output.activate();

                training_result_input.for_each(|time, incoming_data| {
                    assert!(incoming_data.len() == 1);
                    let training_result = incoming_data.pop().unwrap();

                    match training_result_stash.entry(time.time().clone()) {
                        Occupied(_entry) => {
                            panic!("Received more than one training result per timestamp")
                        }
                        Vacant(entry) => {
                            debug!("W{}: Saved training result at {:?}", worker, entry.key());
                            entry.insert(training_result);
                        }
                    }
                });

                training_data_input.for_each(|time, incoming_data| {
                    training_data_stash
                        .entry(time.retain())
                        .or_insert_with(Vec::new)
                        .extend(incoming_data.drain(..));
                });

                for (time, training_data_vec) in &mut training_data_stash {
                    if frontiers.iter().all(|f| !f.less_equal(time)) {
                        let training_result = training_result_stash
                            .remove(time)
                            .expect("retrieve training result for corresponding data");

                        let mut residuals_session = residuals_handle.session(time);

                        if time.inner + 1 >= model.iterations {
                            let boost_chain = boost_chain_stash
                                .remove(&time.time().outer.clone())
                                .unwrap();
                            // send boost chain
                            training_data_vec.clear();
                            info!("Completed boosting, returning accumulated model");
                            boost_chain_handle.session(time).give(boost_chain);
                        } else {
                            let tmp_chain = boost_chain_stash
                                .entry(time.time().outer.clone())
                                .or_insert_with(BoostChain::default);

                            let total_items = training_data_vec.iter().map(|td| td.y().len()).sum();
                            let mut histogram =
                                Histogram::<L, L>::new(<usize>::min(100_000, total_items));

                            // calculate residuals and send to next iteration
                            let data: Vec<_> = training_data_vec
                                .drain(..)
                                .map(|td| {
                                    let previous_predictions = tmp_chain.predict_samples(&td.x());
                                    let added_predictions =
                                        training_result.predict_samples(&td.x());

                                    (
                                        td,
                                        previous_predictions.expect("Predict items"),
                                        added_predictions.expect("Predict items"),
                                    )
                                })
                                .collect();

                            for (training_data, previous_predictions, added_predictions) in &data {
                                Zip::from(&training_data.y())
                                    .and(previous_predictions.view())
                                    .and(added_predictions.view())
                                    .apply(|&actual, &previous, &added| {
                                        let diff = (actual - previous) / added;
                                        let weight = added.abs();
                                        histogram.insert(diff, weight);
                                    });
                            }

                            let multi = histogram.median().expect("get median");
                            tmp_chain.push_item(multi, training_result);

                            for (mut training_data, previous_predictions, added_predictions) in data {
                                let previous_predictions: Array1<L> = previous_predictions.into();
                                let added_predictions: Array1<L> = added_predictions.into();
                                let combined_predictions: Array1<L> =
                                    previous_predictions + added_predictions * multi;

                                // for LAD, the gradients are the sign of the difference between the actual
                                // and predicted values
                                training_data.y = (&training_data.y()
                                    - &combined_predictions.view())
                                    //.map(|x| x.signum())
                                    .into();

                                info!("Calculated residuals and sending on to next iteration");
                                info!("{:?}", training_data.y());
                                residuals_session.give(training_data);
                            }
                        }
                    }
                }

                training_data_stash.retain(|_time, item| !item.is_empty());
            }
        });

        (residuals_stream, boost_chain_stream.leave())
    }
}

pub trait ResidualLossFunction<L> {
    fn loss_gradients(predicted: &ArrayView1<L>, actual: &ArrayView1<L>) -> Array1<L>;
    fn optimize_stage_multiplier(
        actual: &ArrayView1<L>,
        previous_predictions: &ArrayView1<L>,
        added_predictions: &ArrayView1<L>,
    ) -> L;
}
