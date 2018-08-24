use super::boost_chain::BoostChain;
use super::GradientBoostingRegression;
use data::serialization::{AsView, AbomonableArray1};
use data::TrainingData;
use fnv::FnvHashMap;
use models::decision_tree::histogram_generics::ContinuousValue;
use models::LabelingModelAttributes;
use models::PredictSamples;
use ndarray::prelude::*;
use ndarray::{ScalarOperand, Zip};
use num_traits::FromPrimitive;
use num_traits::NumAssign;
use num_traits::ToPrimitive;
use std::collections::hash_map::Entry::*;
use std::fmt::Debug;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::{generic::builder_rc::OperatorBuilder, *};
use timely::dataflow::{scopes::Child, Scope, Stream};
use timely::{Data, ExchangeData};

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
        _model: GradientBoostingRegression<InnerModel, T, L>,
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
