use self::boost_chain::BoostChain;
use self::gradient_vectors::CalculateResiduals;
use data::dataflow::{ApplyLatest, CombineEachTime};
use data::serialization::*;
use data::TrainingData;
use failure::Fail;
use models::decision_tree::histogram_generics::ContinuousValue;
use models::*;
use ndarray::prelude::*;
use ndarray::ScalarOperand;
use num_traits::{FromPrimitive, NumAssign, ToPrimitive, Zero};
use std::fmt::Debug;
use std::marker::PhantomData;
use timely::dataflow::operators::generic::source;
use timely::dataflow::scopes::Child;
use timely::dataflow::{operators::*, Scope, Stream};
use timely::{Data, ExchangeData};

mod boost_chain;
mod gradient_vectors;

#[derive(Clone, Abomonation)]
pub struct GradientBoostingRegression<InnerModel, T, L> {
    iterations: u64,
    inner_model: InnerModel,
    learning_rate: L,
    _t: PhantomData<T>,
}

impl<InnerModel, T, L> GradientBoostingRegression<InnerModel, T, L> {
    pub fn new(iterations: u64, inner_model: InnerModel, learning_rate: L) -> Self {
        GradientBoostingRegression {
            iterations,
            inner_model,
            learning_rate,
            _t: PhantomData,
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

            let chain_initializer = source(boost_iter_scope, "InitBoosting", |cap| {
                let mut cap = Some(cap);
                move |output| {
                    if let Some(cap) = cap.take() {
                        if worker == 0 {
                            output
                                .session(&cap)
                                .give(BoostChain::<InnerModel, T, L>::new(vec![], learning_rate));
                        }
                    }
                }
            });

            let (boost_chain_stream, final_out) = chain_initializer
                .concat(&chain_cycle)
                .branch_when(move |time| time.inner >= iterations);

            let training_data = self.enter(boost_iter_scope);

            let residuals_out = training_data
                .concat(&residuals_cycle)
                .inspect_time(move |time, _| {
                    debug!("W{}: Received residuals (round {})", worker, time.inner)
                })
                .calculate_residuals(model.clone(), &boost_chain_stream, &training_data);

            residuals_out.connect_loop(residuals_loop_handle);
            residuals_out.train(&model.inner_model)
                .inspect_time(move |time, _| {
                    debug!(
                        "W{}: Completed training model to residuals (round {})",
                        worker, time.inner
                    )
                })
                .combine_each_time(
                    &boost_chain_stream,
                    |learner_result_vec, boost_chain_vec| {
                        learner_result_vec
                            .drain(..)
                            .zip(boost_chain_vec.drain(..))
                            .map(|(result, mut chain)| {
                                chain.push_item(L::one(), result);
                                chain
                            })
                            .collect()
                    },
                )
                .connect_loop(chain_loop_handle);

            final_out.leave()
        })
    }
}
