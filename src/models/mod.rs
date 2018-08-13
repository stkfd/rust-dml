use timely::ExchangeData;
use data::providers::{DataSourceSpec, IndexableData};
use failure::{Error, Fail};
use timely::dataflow::{Scope, Stream};
use timely::Data;

pub mod decision_tree;
pub mod gradient_boost;
pub mod kmeans;

#[derive(Fail, Debug, Abomonation, Clone)]
pub enum ModelError<Inner: Data + Fail> {
    #[fail(display = "Prediction failed: {}", _0)]
    PredictionFailed(#[cause] Inner),
}

pub trait ModelAttributes: ExchangeData {
    type UnlabeledSamples: Data;
    type TrainingResult: Data;
}

pub trait SupModelAttributes: ModelAttributes {
    type LabeledSamples: Data;
    type Predictions: Data;
    type PredictErr: Fail + Data;
}

pub trait Train<S: Scope, M: ModelAttributes> {
    fn train(&self, model: &M) -> Stream<S, M::TrainingResult>;
}

pub trait TrainMeta<S: Scope, M: ModelAttributes> {
    fn train_meta(&self, model: &M) -> Stream<S, M::TrainingResult>;
}

pub trait Predict<S: Scope, M: SupModelAttributes, E: Data + Fail> {
    fn predict(
        &self,
        model: &M,
        train_result: Stream<S, M::TrainingResult>,
    ) -> Stream<S, Result<M::Predictions, ModelError<E>>>;
}

pub trait PredictSamples<Samples, Predictions, E: Data + Fail> {
    fn predict_samples(&self, input: &Samples) -> Result<Predictions, ModelError<E>>;
}

pub trait UnSupModel<Inputs: IndexableData, Predictions, TrainingOutput> {
    /// Predict output from inputs.
    fn predict<S: Scope, Source: DataSourceSpec<Inputs>>(
        &mut self,
        scope: &mut S,
        inputs: Source,
    ) -> Result<Stream<S, Predictions>, Error>;

    /// Train the model using inputs.
    fn train<S: Scope, Source: DataSourceSpec<Inputs>>(
        &mut self,
        scope: &mut S,
        inputs: Source,
    ) -> Result<Stream<S, TrainingOutput>, Error>;
}
