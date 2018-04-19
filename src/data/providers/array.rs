use data::providers::{DataSource, DataSourceSpec, IntSliceIndex};
use data::serialization::*;
use ndarray::prelude::*;
use std::convert::TryFrom;
use timely::Data;
use Result;

/// Creates a data provider from a two-dimensional array.
/// Mainly used for testing and examples.
#[derive(Clone, Abomonation)]
pub struct ArrayProviderSpec<T: Data> {
    array: AbomonableArray2<T>,
}

impl<T: Data> ArrayProviderSpec<T> {
    pub fn new(array: Array2<T>) -> Self {
        ArrayProviderSpec {
            array: array.into(),
        }
    }
}

impl<T: Data + Copy> DataSourceSpec<AbomonableArray2<T>> for ArrayProviderSpec<T> {
    type Provider = ArrayProvider<T>;
}

impl<T: Data> TryFrom<ArrayProviderSpec<T>> for ArrayProvider<T> {
    type Error = ::failure::Error;

    fn try_from(from: ArrayProviderSpec<T>) -> Result<Self> {
        Ok(ArrayProvider {
            array: from.array.into(),
        })
    }
}

/// Data provider that uses a two-dimensional array as its internal
/// data source. Mainly used for testing and examples.
pub struct ArrayProvider<T: Data> {
    array: Array2<T>,
}

impl<T: Data + Copy> DataSource<AbomonableArray2<T>> for ArrayProvider<T> {
    /// Fetch a partition of the items in this `DataSource`
    fn slice(&mut self, idx: IntSliceIndex<usize>) -> Result<AbomonableArray2<T>> {
        Ok(self.array
            .slice(s![idx.start..(idx.start + idx.length), ..])
            .to_owned()
            .into())
    }

    /// Select specific rows from the data source
    fn select(&mut self, indices: &[usize]) -> Result<AbomonableArray2<T>> {
        Ok(self.array.select(Axis(0), indices).into())
    }

    /// Iterate over indices to all the item chunks of the given size in this `DataSource`
    fn chunk_indices(
        &mut self,
        chunk_length: usize,
    ) -> Result<Box<Iterator<Item = IntSliceIndex<usize>>>> {
        let len = self.array.len_of(Axis(0));
        let num_chunks = ((len as f64) / (chunk_length as f64)).ceil() as usize;

        Ok(Box::new((0..num_chunks).map(move |i| {
            let start = i * chunk_length;
            if start + chunk_length < len {
                IntSliceIndex::new(start, chunk_length)
            } else {
                IntSliceIndex::new(start, len - start)
            }
        })))
    }

    /// Total number of available items
    fn count(&mut self) -> Result<usize> {
        Ok(self.array.len_of(Axis(0)))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use data::providers::IntSliceIndex;
    use std::convert::TryInto;

    #[test]
    fn indices() {
        let a = arr2(&[
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]);

        let spec = ArrayProviderSpec::new(a);
        let mut provider: ArrayProvider<_> = spec.try_into().unwrap();
        let chunk_indices = provider.chunk_indices(4).unwrap().collect::<Vec<_>>();
        assert_eq!(
            chunk_indices,
            vec![IntSliceIndex::new(0, 4), IntSliceIndex::new(4, 2)]
        )
    }

    #[test]
    fn slice() {
        let a = arr2(&[
            [1, 1, 1],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [1, 1, 1],
            [1, 1, 1],
        ]);

        let spec = ArrayProviderSpec::new(a);
        let mut provider: ArrayProvider<_> = spec.try_into().unwrap();
        let slice: Array2<_> = provider.slice(IntSliceIndex::new(2, 2)).unwrap().into();
        assert_eq!(slice, arr2(&[[2, 2, 2], [3, 3, 3]]))
    }
}
