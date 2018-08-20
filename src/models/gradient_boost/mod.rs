use data::dataflow::ApplyLatest;
use data::serialization::*;
use data::TrainingData;
use failure::Fail;
use fnv::FnvHashMap;
use models::decision_tree::histogram_generics::{BaseHistogram, ContinuousValue, Median};
use models::decision_tree::regression::histogram::Histogram;
use models::*;
use ndarray::prelude::*;
use ndarray::{ScalarOperand, Zip};
use num_traits::{Float, FromPrimitive, NumAssign, ToPrimitive, Zero};
use std::collections::hash_map::Entry::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::generic::builder_rc::OperatorBuilder;
use timely::dataflow::operators::generic::source;
use timely::dataflow::scopes::Child;
use timely::dataflow::{operators::*, Scope, Stream};
use timely::progress::nested::product::Product;
use timely::{Data, ExchangeData};

#[derive(Clone, Abomonation)]
pub struct GradientBoostingRegression<InnerModel, T, L> {
    iterations: u64,
    inner_model: InnerModel,
    learning_rate: L,
    _t: PhantomData<T>,
    _l: PhantomData<L>,
}

impl<InnerModel, T, L> GradientBoostingRegression<InnerModel, T, L> {
    pub fn new(iterations: u64, inner_model: InnerModel, learning_rate: L) -> Self {
        GradientBoostingRegression {
            iterations,
            inner_model,
            learning_rate,
            _t: PhantomData,
            _l: PhantomData,
        }
    }
}

impl<InnerModel: LabelingModelAttributes, T: ExchangeData, L: ExchangeData> ModelAttributes
    for GradientBoostingRegression<InnerModel, T, L>
{
    type TrainingResult = BoostChain<InnerModel, T, L>;
}

impl<InnerModel: LabelingModelAttributes, T: ExchangeData, L: ExchangeData> LabelingModelAttributes
    for GradientBoostingRegression<InnerModel, T, L>
{
    type Predictions = AbomonableArray1<L>;
    type PredictErr = InnerModel::PredictErr;
}

#[derive(Clone, Abomonation, Debug)]
pub struct BoostChain<InnerModel: LabelingModelAttributes, T, L> {
    chain: Vec<(L, InnerModel::TrainingResult)>,
    learning_rate: L,
    phantom: PhantomData<(InnerModel, T, L)>,
}

impl<InnerModel: LabelingModelAttributes, T, L: Float> BoostChain<InnerModel, T, L> {
    pub fn new(
        chain: Vec<(L, InnerModel::TrainingResult)>,
        learning_rate: L,
    ) -> BoostChain<InnerModel, T, L> {
        BoostChain {
            chain,
            learning_rate,
            phantom: PhantomData,
        }
    }

    pub fn push_item(&mut self, item: InnerModel::TrainingResult) {
        self.chain.push((L::zero(), item))
    }
}

impl<A, InnerModel, T, L> PredictSamples<A, AbomonableArray1<L>, InnerModel::PredictErr>
    for BoostChain<InnerModel, T, L>
where
    for<'a> &'a A: AsArray<'a, T, Ix2>,
    L: Float + ScalarOperand,
    InnerModel: LabelingModelAttributes,
    for<'a> InnerModel::TrainingResult:
        PredictSamples<ArrayView2<'a, T>, AbomonableArray1<L>, InnerModel::PredictErr>,
{
    fn predict_samples(
        &self,
        a: &A,
    ) -> Result<AbomonableArray1<L>, ModelError<InnerModel::PredictErr>> {
        let view = a.into();
        let mut agg = Array1::zeros(view.rows());

        for (multi, training_output) in &self.chain {
            let prediction: Array1<L> = training_output.predict_samples(&view)?.into();
            agg = agg + prediction * self.learning_rate;
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
    InnerModel: LabelingModelAttributes,
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
    InnerModel: LabelingModelAttributes<Predictions = AbomonableArray1<L>>,
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
        let learning_rate = model.learning_rate;
        let mut scope = self.scope();
        let worker = scope.index();

        scope.scoped::<u64, _, _>(|boost_iter_scope| {
            let iterations = model.iterations;
            let (chain_loop_handle, chain_cycle) = boost_iter_scope.loop_variable(iterations, 1);
            let (residuals_loop_handle, residuals_cycle) =
                boost_iter_scope.loop_variable(iterations - 1, 1);

            let (boost_chain_stream, final_out) = source(boost_iter_scope, "InitBoosting", |cap| {
                let mut cap = Some(cap);
                move |output| {
                    if let Some(cap) = cap.take() {
                        output
                            .session(&cap)
                            .give(BoostChain::<InnerModel, T, L>::new(vec![], learning_rate));
                    }
                }
            }).filter(move |_| worker == 0)
                .concat(&chain_cycle)
                .branch_when(move |time| time.inner >= iterations);

            let training_data = self.enter(boost_iter_scope);

            let residuals_out = training_data
                .concat(&residuals_cycle)
                .inspect_time(move |time, _| {
                    debug!("W{}: Received residuals (round {})", worker, time.inner)
                })
                .calculate_residuals(model.clone(), &boost_chain_stream, &training_data);

            <Stream<_, _> as Train<_, InnerModel>>::train(&residuals_out, &model.inner_model)
                .inspect_time(move |time, _| {
                    debug!(
                        "W{}: Completed training model to residuals (round {})",
                        worker, time.inner
                    )
                })
                .binary_frontier(
                    &boost_chain_stream,
                    Pipeline,
                    Pipeline,
                    "ExtendBoostChain",
                    |_, _| {
                        let mut stash = FnvHashMap::default();
                        move |learner_result_input, boost_chain_input, output| {
                            learner_result_input.for_each(|time, data| {
                                let (result_vec, _boost_chain_vec) = stash
                                    .entry(time.retain())
                                    .or_insert_with(|| (vec![], vec![]));
                                result_vec.extend(data.drain(..));
                            });
                            boost_chain_input.for_each(|time, data| {
                                let (_result_vec, boost_chain_vec) = stash
                                    .entry(time.retain())
                                    .or_insert_with(|| (vec![], vec![]));
                                boost_chain_vec.extend(data.drain(..));
                            });

                            let frontiers = &[
                                learner_result_input.frontier(),
                                boost_chain_input.frontier(),
                            ];
                            for (cap, (learner_result_vec, boost_chain_vec)) in &mut stash {
                                if frontiers.iter().all(|f| !f.less_equal(cap.time())) {
                                    let mut session = output.session(&cap);
                                    for (result, mut chain) in
                                        learner_result_vec.drain(..).zip(boost_chain_vec.drain(..))
                                    {
                                        chain.push_item(result);
                                        session.give(chain);
                                    }
                                }
                            }
                            stash.retain(|time, entry| !entry.0.is_empty() || !entry.1.is_empty());
                        }
                    },
                )
                .connect_loop(chain_loop_handle);
            residuals_out.connect_loop(residuals_loop_handle);

            /*let data = self
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
            boost_chain_stream*/
            final_out.leave()
        })
    }
}

#[allow(type_complexity)]
pub trait CalculateResiduals<
    'a,
    S: Scope,
    Model,
    TrainingData: Data,
    InnerModel: LabelingModelAttributes,
    T,
    L,
>
{
    fn calculate_residuals(
        &self,
        model: Model,
        boost_chain: &Stream<Child<'a, S, u64>, BoostChain<InnerModel, T, L>>,
        original_training_data: &Stream<Child<'a, S, u64>, TrainingData>,
    ) -> Stream<Child<'a, S, u64>, TrainingData>;
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
    InnerModel: LabelingModelAttributes<Predictions = AbomonableArray1<L>>,
    for<'b> InnerModel::TrainingResult:
        ExchangeData
            + Debug
            + PredictSamples<ArrayView2<'b, T>, AbomonableArray1<L>, InnerModel::PredictErr>,
{
    fn calculate_residuals(
        &self,
        model: GradientBoostingRegression<InnerModel, T, L>,
        boost_chain: &Stream<Child<'a, S, u64>, BoostChain<InnerModel, T, L>>,
        original_training_data: &Stream<Child<'a, S, u64>, TrainingData<T, L>>,
    ) -> Stream<Child<'a, S, u64>, TrainingData<T, L>> {
        let worker = self.scope().index();
        let mut builder = OperatorBuilder::new("CalculateResiduals".to_owned(), self.scope());

        let mut residuals_input = builder.new_input(self, Pipeline);
        let mut training_data_input = builder.new_input(original_training_data, Pipeline);
        let mut boost_chain_input = builder.new_input(&boost_chain.broadcast(), Pipeline);

        let (mut residuals_output, residuals_stream) = builder.new_output();

        builder.build(|_fsdd| {
            let mut training_data_stash = FnvHashMap::default();
            let mut residuals_stash = FnvHashMap::default();
            let mut boost_chain_stash = FnvHashMap::default();

            move |frontiers| {
                let mut residuals_handle = residuals_output.activate();

                boost_chain_input.for_each(|time, incoming_data| {
                    assert!(incoming_data.len() == 1);
                    let training_result = incoming_data.pop().unwrap();

                    match boost_chain_stash.entry(time.time().clone()) {
                        Occupied(_entry) => {
                            panic!("Received more than one boost chain per timestamp")
                        }
                        Vacant(entry) => {
                            debug!("W{}: Saved training result at {:?}", worker, entry.key());
                            entry.insert(training_result);
                        }
                    }
                });

                residuals_input.for_each(|time, incoming_data| {
                    debug!("got residuals");
                    residuals_stash
                        .entry(time.retain())
                        .or_insert_with(Vec::new)
                        .extend(incoming_data.drain(..));
                });

                training_data_input.for_each(|time, incoming_data| {
                    debug!("got original td");
                    training_data_stash
                        .entry(time.time().outer.clone())
                        .or_insert_with(Vec::new)
                        .extend(incoming_data.drain(..).map(|td| td.y));
                });

                for (time, residuals_vec) in &mut residuals_stash {
                    if frontiers.iter().all(|f| !f.less_equal(time)) {
                        let boost_chain = boost_chain_stash
                            .remove(time)
                            .expect("retrieve boost chain for corresponding residuals");
                        let original_td_vec = training_data_stash
                            .get(&time.time().outer)
                            .expect("get original training data");

                        /*let total_items = training_data_vec.iter().map(|td| td.y().len()).sum();
                            let mut histogram =
                                Histogram::<L, L>::new(<usize>::min(100_000, total_items));*/

                        // calculate residuals and send to next iteration
                        let mut session = residuals_handle.session(&time);
                        residuals_vec
                            .drain(..)
                            .zip(original_td_vec.iter())
                            .for_each(|(mut residuals, original_td)| {
                                let predictions = boost_chain
                                    .predict_samples(&residuals.x())
                                    .expect("Predict items");
                                debug!("{:?}", predictions.view());
                                Zip::from(residuals.y_mut())
                                    .and(&predictions.view())
                                    .and(&original_td.view())
                                    .apply(|residual, &prediction, &actual| {
                                        *residual = actual - prediction;
                                    });

                                session.give(residuals);
                            });

                        /*for (training_data, previous_predictions, added_predictions) in &data {
                                Zip::from(&training_data.y())
                                    .and(previous_predictions.view())
                                    .and(added_predictions.view())
                                    .apply(|&actual, &previous, &added| {
                                        let diff = (actual - previous) / added;
                                        let weight = added.abs();
                                        histogram.insert(diff, weight);
                                    });
                            }*/

                        //let multi = histogram.median().expect("get median");
                    }
                }
                residuals_stash.retain(|_time, item| !item.is_empty());
            }
        });

        residuals_stream
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
