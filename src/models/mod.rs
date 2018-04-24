use data::providers::{DataSourceSpec, IndexableData};
use timely::dataflow::{Scope, Stream};
use Result;

pub mod kmeans;

// pub trait SupModel<T, U> {
//     /// Predict output from inputs.
//     fn predict(&self, inputs: &T) -> Result<U>;

//     /// Train the model using inputs and targets.
//     fn train(&mut self, inputs: &T, targets: &U) -> Result<()>;
// }

pub trait UnSupModel<T: IndexableData, U> {
    /// Predict output from inputs.
    fn predict<S: Scope, Sp: DataSourceSpec<T>>(&mut self, scope: &mut S, inputs: Sp) -> Result<Stream<S, U>>;

    /// Train the model using inputs.
    fn train<S: Scope, Sp: DataSourceSpec<T>>(&mut self, scope: &mut S, inputs: Sp) -> Result<()>;
}
