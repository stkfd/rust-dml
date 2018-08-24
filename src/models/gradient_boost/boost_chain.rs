use data::serialization::AbomonableArray1;
use models::LabelingModelAttributes;
use models::ModelError;
use models::PredictSamples;
use ndarray::prelude::*;
use ndarray::ScalarOperand;
use num_traits::Float;
use std::marker::PhantomData;

#[derive(Clone, Abomonation, Debug)]
pub struct BoostChain<InnerModel: LabelingModelAttributes, T, L> {
    chain: Vec<(L, InnerModel::TrainingResult)>,
    learning_rate: L,
    phantom: PhantomData<T>,
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
