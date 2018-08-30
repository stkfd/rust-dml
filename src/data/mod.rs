pub mod dataflow;
pub mod providers;
pub mod quantize;
pub mod serialization;

use data::serialization::*;
use ndarray::prelude::*;
use serde::ser::{Serialize, SerializeSeq, SerializeTuple, Serializer};

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

pub struct TrainingDataSample<'a, T: 'a, L: 'a> {
    x: ArrayView1<'a, T>,
    y: &'a L,
}

impl<'a, T: 'a, L: 'a> From<(ArrayView1<'a, T>, &'a L)> for TrainingDataSample<'a, T, L> {
    fn from(from: (ArrayView1<'a, T>, &'a L)) -> Self {
        TrainingDataSample {
            x: from.0,
            y: from.1,
        }
    }
}

impl<'a, T, L> Serialize for TrainingDataSample<'a, T, L>
where
    T: 'a + Serialize,
    L: 'a + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tuple = serializer.serialize_tuple(self.x.len() + 1)?;
        for x_el in self.x {
            tuple.serialize_element(x_el)?;
        }
        tuple.serialize_element(self.y)?;
        tuple.end()
    }
}

impl<T, L> Serialize for TrainingData<T, L>
where
    T: Serialize,
    L: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let x = self.x();
        let y = self.y();
        let mut seq = serializer.serialize_seq(Some(x.rows()))?;
        for (row, y) in x.outer_iter().zip(y.iter()) {
            let sample = TrainingDataSample { x: row, y };
            seq.serialize_element(&sample)?;
        }
        seq.end()
    }
}
