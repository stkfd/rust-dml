pub mod dataflow;
pub mod providers;
pub mod quantize;
pub mod serialization;

use data::serialization::*;
use ndarray::prelude::*;
use serde::de::{SeqAccess, Visitor, Deserialize};
use serde::ser::{Serialize, SerializeSeq, SerializeTuple, Serializer};
use std::marker::PhantomData;

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

struct TrainingDataVisitor<T, L> {
    marker: PhantomData<fn() -> TrainingData<T, L>>,
}

impl<T, L> TrainingDataVisitor<T, L> {
    fn new() -> Self {
        TrainingDataVisitor {
            marker: PhantomData,
        }
    }
}

impl<'de, T, L> Visitor<'de> for TrainingDataVisitor<T, L>
where
    T: Deserialize<'de>,
    L: Deserialize<'de>,
{
    // The type that our Visitor is going to produce.
    type Value = TrainingData<T, L>;

    // Format a message stating what data this Visitor expects to receive.
    fn expecting(&self, formatter: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        formatter.write_str("training data array")
    }

    // Deserialize MyMap from an abstract "map" provided by the
    // Deserializer. The MapAccess input is a callback provided by
    // the Deserializer to let us see each entry in the map.
    fn visit_seq<S>(self, mut access: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let x = Vec::with_capacity(access.size_hint().unwrap_or(0));
        let y = Vec::with_capacity(access.size_hint().unwrap_or(0));

        // While there are entries remaining in the input, add them
        // into our map.
        while let Some(value) = access.next_element()? {
        }

        Ok()
    }
}
