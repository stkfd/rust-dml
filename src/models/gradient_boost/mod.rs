use data::dataflow::ApplyLatest;
use data::serialization::*;
use data::TrainingData;
use failure::Fail;
use fnv::FnvHashMap;
use models::decision_tree::histogram_generics::ContinuousValue;
use models::*;
use ndarray::prelude::*;
use ndarray::{ScalarOperand, Zip};
use num_traits::{Float, FromPrimitive, NumAssign, ToPrimitive, Zero};
use std::fmt::Debug;
use std::marker::PhantomData;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::generic::source;
use timely::dataflow::scopes::Child;
use timely::dataflow::{operators::*, Scope, Stream};
use timely::{Data, ExchangeData};
use self::gradient_vectors::CalculateResiduals;

mod gradient_vectors;
mod extend_boost_chain;

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

impl<InnerModel: LabelingModelAttributes, T, L> BoostChain<InnerModel, T, L> {
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

    pub fn push_item(&mut self, scaling_factor: L, item: InnerModel::TrainingResult) {
        self.chain.push((scaling_factor, item))
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

        for (scaling, training_output) in &self.chain {
            let prediction: Array1<L> = training_output.predict_samples(&view)?.into();
            agg = agg + prediction * self.learning_rate * *scaling;
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
                                        chain.push_item(L::one(), result);
                                        session.give(chain);
                                    }
                                }
                            }
                            stash.retain(|_, entry| !entry.0.is_empty() || !entry.1.is_empty());
                        }
                    },
                )
                .connect_loop(chain_loop_handle);
            residuals_out.connect_loop(residuals_loop_handle);

            final_out.leave()
        })
    }
}
