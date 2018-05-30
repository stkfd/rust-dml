use ndarray::prelude::*;
use data::serialization::*;
use data::providers::{DataSourceSpec, IndexableData};
use timely::dataflow::{Scope, Stream};
use Result;

pub mod kmeans;
pub mod spdt;

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
        scope: &mut S,
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
        scope: &mut S,
        inputs: Stream<S, Inputs>,
    ) -> Result<Stream<S, Predictions>>;

    /// Train the model using inputs.
    fn train<S: Scope>(
        &mut self,
        scope: &mut S,
        inputs: Stream<S, Inputs>,
    ) -> Result<Stream<S, TrainingOutput>>;
}


/// Data structure to hold training data for supervised models.
/// Holds a Matrix with input data and an Array with the
/// associated outputs
#[derive(Clone, Abomonation, Debug)]
pub struct TrainingData<T, L> {
    pub x: AbomonableArray2<T>,
    pub y: AbomonableArray1<L>,
}

impl<T, L> TrainingData<T, L> {
    pub fn x<'a, 'b: 'a>(&'b self) -> ArrayView2<'a, T> {
        self.x.view()
    }
    pub fn x_mut<'a, 'b: 'a>(&'b mut self) -> ArrayViewMut2<'a, T> {
        self.x.view_mut()
    }
    pub fn y<'a, 'b: 'a>(&'b self) -> ArrayView1<'a, L> {
        self.y.view()
    }
    pub fn y_mut<'a, 'b: 'a>(&'b mut self) -> ArrayViewMut1<'a, L> {
        self.y.view_mut()
    }
}