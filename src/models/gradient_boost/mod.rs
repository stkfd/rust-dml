use data::dataflow::ApplyLatest;
use data::serialization::*;
use data::TrainingData;
use failure::Fail;
use fnv::FnvHashMap;
use models::decision_tree::histogram_generics::ContinuousValue;
use models::*;
use ndarray::prelude::*;
use num_traits::Float;
use num_traits::Zero;
use std::collections::hash_map::Entry::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem;
use timely::dataflow::channels::pact::{Exchange, Pipeline};
use timely::dataflow::operators::generic::builder_rc::OperatorBuilder;
use timely::dataflow::scopes::Child;
use timely::dataflow::{operators::*, Scope, Stream};
use timely::{Data, ExchangeData};

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

impl<InnerModel: ModelAttributes, T: Data, L: Data> ModelAttributes
    for GradientBoostingRegression<InnerModel, T, L>
{
    type LabeledSamples = TrainingData<T, L>;
    type UnlabeledSamples = AbomonableArray2<T>;
    type Predictions = AbomonableArray1<L>;
    type TrainingResult = BoostChain<InnerModel, T, L>;

    type PredictErr = InnerModel::PredictErr;
}

#[derive(Clone, Abomonation, Debug)]
pub struct BoostChain<InnerModel: ModelAttributes, T, L>(
    pub Vec<InnerModel::TrainingResult>,
    PhantomData<(InnerModel, T, L)>,
);

impl<InnerModel: ModelAttributes, T, L> BoostChain<InnerModel, T, L> {
    pub fn new(items: Vec<InnerModel::TrainingResult>) -> BoostChain<InnerModel, T, L> {
        BoostChain(items, PhantomData)
    }
}

impl<A, InnerModel, T, L> PredictSamples<A, AbomonableArray1<L>, InnerModel::PredictErr>
    for BoostChain<InnerModel, T, L>
where
    for<'a> &'a A: AsArray<'a, T, Ix2>,
    L: Clone + Zero,
    InnerModel: ModelAttributes,
    for<'a> InnerModel::TrainingResult:
        PredictSamples<ArrayView2<'a, T>, AbomonableArray1<L>, InnerModel::PredictErr>,
{
    fn predict_samples(
        &self,
        a: &A,
    ) -> Result<AbomonableArray1<L>, ModelError<InnerModel::PredictErr>> {
        let view = a.into();
        let mut agg = Array1::zeros(view.rows());

        for training_output in &self.0 {
            let prediction: Array1<L> = training_output.predict_samples(&view)?.into();
            agg = agg + prediction;
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
    InnerModel: ModelAttributes,
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
    L: ContinuousValue,
    InnerModel: ModelAttributes<Predictions = AbomonableArray1<L>>,
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
        scope.scoped::<u64, _, _>(|boost_iter_scope| {
            let (loop_handle, cycle) = boost_iter_scope.loop_variable(model.iterations, 1);
            let data = self.enter(boost_iter_scope).concat(&cycle);

            let training_results = <Stream<Child<'_, _, u64>, _> as Train<_, InnerModel>>::train(
                &data,
                &model.inner_model,
            );

            let (residuals_stream, boost_chain_stream) =
                data.calculate_residuals(model.clone(), &training_results, AbsoluteLoss);

            residuals_stream.connect_loop(loop_handle);
            boost_chain_stream
        })
    }
}

pub trait CalculateResiduals<
    'a,
    S: Scope,
    Model,
    TrainingData: Data,
    InnerModel: ModelAttributes,
    LossFunc,
    T,
    L,
>
{
    fn calculate_residuals(
        &self,
        model: Model,
        training_results: &Stream<Child<'a, S, u64>, InnerModel::TrainingResult>,
        residual_calc: LossFunc,
    ) -> (
        Stream<Child<'a, S, u64>, TrainingData>,
        Stream<S, BoostChain<InnerModel, T, L>>,
    );
}

impl<'a, S, T, L, InnerModel, LossFunc>
    CalculateResiduals<
        'a,
        S,
        GradientBoostingRegression<InnerModel, T, L>,
        TrainingData<T, L>,
        InnerModel,
        LossFunc,
        T,
        L,
    > for Stream<Child<'a, S, u64>, TrainingData<T, L>>
where
    T: ExchangeData + Debug,
    L: ExchangeData + Debug,
    S: Scope,
    InnerModel: ModelAttributes<Predictions = AbomonableArray1<L>>,
    for<'b> InnerModel::TrainingResult:
        ExchangeData
            + Debug
            + PredictSamples<ArrayView2<'b, T>, AbomonableArray1<L>, InnerModel::PredictErr>,
    LossFunc: ResidualLossFunction<L>,
{
    fn calculate_residuals(
        &self,
        model: GradientBoostingRegression<InnerModel, T, L>,
        training_results: &Stream<Child<'a, S, u64>, InnerModel::TrainingResult>,
        _residual_calc: LossFunc,
    ) -> (
        Stream<Child<'a, S, u64>, TrainingData<T, L>>,
        Stream<S, BoostChain<InnerModel, T, L>>,
    ) {
        let worker = self.scope().index();
        let mut builder = OperatorBuilder::new("Branch".to_owned(), self.scope());

        let mut training_data_input = builder.new_input(self, Exchange::new(|_| 0_u64));
        let mut training_result_input =
            builder.new_input(training_results, Exchange::new(|_| 0_u64));

        let (mut residuals_output, residuals_stream) = builder.new_output();
        let (mut boost_chain_output, boost_chain_stream) = builder.new_output();

        builder.build(|_fsdd| {
            let mut training_data_stash = FnvHashMap::default();
            let mut training_result_stash = FnvHashMap::default();
            
            move |frontiers| {
                let mut residuals_handle = residuals_output.activate();
                let mut boost_chain_handle = boost_chain_output.activate();

                training_result_input.for_each(|time, incoming_data| {
                    assert!(incoming_data.len() == 1);
                    let training_result = incoming_data.pop().unwrap();
                    match training_result_stash.entry(time.retain()) {
                        Occupied(_entry) => {
                            panic!("Received more than one training result per timestamp")
                        }
                        Vacant(entry) => {
                            debug!("W{}: Saved training result at {:?}", worker, entry.key().time());
                            entry.insert((Some(training_result), Vec::new()));
                        }
                    }
                });

                training_data_input.for_each(|time, incoming_data| {
                    debug!("W{}: Incoming training data at {:?}", worker, time.time());
                    training_data_stash
                        .entry(time.retain())
                        .or_insert_with(Vec::new)
                        .extend(incoming_data.drain(..));
                });

                for (time, training_data_vec) in &mut training_data_stash {
                    if frontiers.iter().all(|f| !f.less_equal(time)) {
                        debug!("W{}: {:?}", worker, time.time());
                        debug!("W{}: {:?}", worker, training_result_stash);
                        let (training_result_opt, temp_boost_chain) = training_result_stash
                            .get_mut(time)
                            .expect("retrieve training result for corresponding data");

                        let training_result = training_result_opt.take().unwrap();
                        temp_boost_chain.push(training_result.clone());

                        let mut residuals_session = residuals_handle.session(time);

                        if time.inner >= model.iterations {
                            // send boost chain
                            let boost_chain = BoostChain::<InnerModel, T, L>::new(mem::replace(
                                temp_boost_chain,
                                Vec::new(),
                            ));
                            boost_chain_handle.session(time).give(boost_chain);
                        } else {
                            // calculate residuals and send to next iteration
                            for mut training_data in training_data_vec.drain(..) {
                                match training_result.predict_samples(&training_data.x()) {
                                    Ok(predictions) => {
                                        let predictions: Array1<L> = predictions.into();
                                        let loss = LossFunc::residual_loss(
                                            predictions,
                                            &training_data.y(),
                                        );
                                        training_data.y = loss.into();
                                        debug!("Calculated residuals and sending on to next iteration");
                                        residuals_session.give(training_data);
                                    }
                                    Err(err) => unimplemented!(),
                                }
                            }
                        }
                    }
                }

                training_data_stash.retain(|_time, item| !item.is_empty());
                training_result_stash.retain(|_time, (result_opt, tmp_chain)| {
                    result_opt.is_some() || !tmp_chain.is_empty()
                });
            }
        });

        (residuals_stream, boost_chain_stream.leave())
    }
}

pub trait ResidualLossFunction<L> {
    fn residual_loss(predicted: Array1<L>, actual: &ArrayView1<L>) -> Array1<L>;
}

#[derive(Clone, Copy, Debug)]
pub struct AbsoluteLoss;

impl<L: Clone + Float> ResidualLossFunction<L> for AbsoluteLoss {
    fn residual_loss(predicted: Array1<L>, actual: &ArrayView1<L>) -> Array1<L> {
        (predicted - actual).mapv_into(|x| x.abs())
    }
}
