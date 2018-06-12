pub mod dataflow;
pub mod providers;
pub mod serialization;

use data::serialization::*;
use ndarray::prelude::*;

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
