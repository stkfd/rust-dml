use data::providers::{DataSourceSpec, IndexableData};
use timely::dataflow::{Scope, Stream};
use Result;

pub mod kmeans;
pub mod decision_tree;

//pub trait SupModel<T, U> {
//    /// Predict output from inputs.
//    fn predict<S: Scope, Source: DataSourceSpec<Inputs>>(&mut self, scope: &mut S, inputs: Source) -> Result<Stream<S, Predictions>>;
//
//    /// Train the model using inputs and targets.
//    fn train(&mut self, inputs: &T, targets: &U) -> Result<()>;
//}

pub trait StreamingSupModel<
    TrainingInput,
    PredictionInput,
    Predictions,
    TrainingOutput,
>
{
    /// Predict output from inputs.
    fn predict<S: Scope>(
        &mut self,
        training_results: Stream<S, TrainingOutput>,
        inputs: Stream<S, PredictionInput>,
    ) -> Result<Stream<S, Predictions>>;

    /// Train the model using inputs and targets.
    fn train<S: Scope>(
        &mut self,
        scope: &mut S,
        data: Stream<S, TrainingInput>,
    ) -> Result<Stream<S, TrainingOutput>>;
}

pub trait UnSupModel<Inputs: IndexableData, Predictions, TrainingOutput> {
    /// Predict output from inputs.
    fn predict<S: Scope, Source: DataSourceSpec<Inputs>>(
        &mut self,
        scope: &mut S,
        inputs: Source,
    ) -> Result<Stream<S, Predictions>>;

    /// Train the model using inputs.
    fn train<S: Scope, Source: DataSourceSpec<Inputs>>(
        &mut self,
        scope: &mut S,
        inputs: Source,
    ) -> Result<Stream<S, TrainingOutput>>;
}

pub trait StreamingUnSupModel<Inputs: IndexableData, Predictions, TrainingOutput> {
    /// Predict output from inputs.
    fn predict<S: Scope>(
        &mut self,
        training_results: Stream<S, TrainingOutput>,
        inputs: Stream<S, Inputs>,
    ) -> Result<Stream<S, Predictions>>;

    /// Train the model using inputs.
    fn train<S: Scope>(
        &mut self,
        scope: &mut S,
        inputs: Stream<S, Inputs>,
    ) -> Result<Stream<S, TrainingOutput>>;
}
