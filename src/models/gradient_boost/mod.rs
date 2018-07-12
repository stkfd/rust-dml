use data::serialization::*;
use data::TrainingData;
use models::decision_tree::histogram_generics::ContinuousValue;
use ndarray::prelude::*;
use std::iter::Sum;
use std::marker::PhantomData;
use timely::dataflow::{Scope, Stream, operators::*};
use timely::Data;

pub struct GradientBoostingRegression<InnerModel> {
    iterations: u64,
    inner_model: InnerModel,
}

#[derive(Clone, Abomonation)]
pub struct BoostedModelChain<InnerModel, TrainingOutput>(
    pub Vec<TrainingOutput>,
    PhantomData<InnerModel>,
);

impl<InnerModel, TrainingOutput> BoostedModelChain<InnerModel, TrainingOutput> {
    pub fn predict_samples<T, L>(&self, samples: AbomonableArray2<T>) -> AbomonableArray1<L> {
        unimplemented!()
    }
}

/*
impl<T, L, InnerModel, InnerTrainingOutput>
    StreamingSupModel<
        TrainingData<T, L>,
        AbomonableArray2<T>,
        AbomonableArray1<L>,
        BoostedModelChain<InnerModel, InnerTrainingOutput>,
    > for GradientBoostingRegression<InnerModel>
where
    T: Data,
    L: ContinuousValue + Sum,
    InnerTrainingOutput: Data,
    InnerModel: Data
        + StreamingSupModel<
            TrainingData<T, L>,
            AbomonableArray2<T>,
            AbomonableArray1<L>,
            InnerTrainingOutput,
        >,
{
    fn predict_samples(
        samples: &AbomonableArray2<T>,
        model_chain: &BoostedModelChain<InnerModel, InnerTrainingOutput>,
    ) -> AbomonableArray1<L> {
        model_chain
            .0
            .iter()
            .map(|training_output| InnerModel::predict_samples(samples, training_output).into())
            .fold(
                Array1::zeros(samples.view().rows()),
                |agg, predictions: Array1<L>| agg + predictions,
            )
            .into()
    }

    /// Train the model using inputs and targets.
    fn train<S: Scope>(
        &mut self,
        scope: &mut S,
        data: Stream<S, TrainingData<T, L>>,
    ) -> Result<Stream<S, BoostedModelChain<InnerModel, InnerTrainingOutput>>> {
        scope.scoped::<u64,_,_>(|boost_iter_scope| {
            let (loop_handle, cycle) = boost_iter_scope.loop_variable(self.iterations, 1);
            let data = data.enter(boost_iter_scope);

            self.inner_model.train(boost_iter_scope, data).expect("train model")
        });
        unimplemented!()
    }
}
*/