use data::serialization::AbomonableArray2;
use num_traits::PrimInt;
use std::convert::TryFrom;
use timely::{Data, ExchangeData};
use Result;

pub mod operators;
pub mod csv;
pub mod array;

/// Trait representing a collection of data that can be retrieved from a `DataSource`. The collection
/// has an associated index type (`ItemIndex`) and be partitioned into chunks. Individual chunks are
/// addressed with the associated `SliceIndex` type. A chunk of data can be retrieved from a `DataSource`
/// by passing it a `SliceIndex` which addresses it.
pub trait IndexableData: Data {
    type SliceIndex: IndexesSlice + ExchangeData;
}

impl <T: Data> IndexableData for AbomonableArray2<T> {
    type SliceIndex = IntSliceIndex<usize>;
}

impl <T: Data> IndexableData for Vec<T> {
    type SliceIndex = IntSliceIndex<usize>;
}

/// Specifies the data needed to create a `DataSource` in a serializable struct that can be sent to
/// other workers. Types that implement this trait can be converted into their respective
/// `DataSource` using `try_from`/`try_into`.
pub trait DataSourceSpec<Collection: IndexableData>: Data {
    type Provider: TryFrom<Self, Error = ::failure::Error> + DataSource<Collection>;

    fn to_provider(&self) -> Result<Self::Provider> {
        <Self::Provider>::try_from(self.clone())
    }

    fn into_provider(self) -> Result<Self::Provider> {
        <Self::Provider>::try_from(self)
    }
}

pub trait DataSource<Collection: IndexableData> {
    /// Fetch a partition of the items in this `DataSource`
    fn slice(&mut self, idx: Collection::SliceIndex) -> Result<Collection>;

    /// Fetch the complete contents of the data source
    fn all(&mut self) -> Result<Collection>;

    /// Select specific rows from the data source
    fn select(&mut self, indices: &[<Collection::SliceIndex as IndexesSlice>::Idx]) -> Result<Collection>;

    /// Iterate over indices to all the item chunks of the given size in this `DataSource`
    fn chunk_indices(
        &mut self,
        chunk_length: usize,
    ) -> Result<Box<Iterator<Item = Collection::SliceIndex>>>;

    /// Total number of available items
    fn count(&mut self) -> Result<usize>;
}

pub trait IndexesSlice {
    type Idx: ExchangeData;

    /// Convert an Index relative to the start of this Slice to an absolute index that points to the
    /// item in the whole Dataset
    fn absolute_index(&self, index_in_slice: Self::Idx) -> Self::Idx;
}

#[derive(Abomonation, Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntSliceIndex<T: PrimInt> {
    pub start: T,
    pub length: T,
}

impl<T: ExchangeData + PrimInt> IndexesSlice for IntSliceIndex<T> {
    type Idx = T;

    #[inline]
    fn absolute_index(&self, index_in_slice: T) -> T {
        self.start + index_in_slice
    }
}

impl<T: PrimInt> IntSliceIndex<T> {
    pub fn new(start: T, length: T) -> Self {
        IntSliceIndex { start, length }
    }
}
