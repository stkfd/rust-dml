use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use models::decision_tree::tree::NodeIndex;
use num_traits::Float;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::Sum;
use timely::ExchangeData;

mod btree;
mod fnv;
mod vec;

pub use self::btree::{BTreeHistogramSet, SerializableBTreeHistogramSet};
pub use self::fnv::{FnvHistogramSet, SerializableFnvHistogramSet};
pub use self::vec::{SerializableVecHistogramSet, VecHistogramSet};


pub trait HistogramSet<K, H: HistogramSetItem>: Default {
    fn get(&self, key: &K) -> Option<&H>;
    fn get_mut(&mut self, key: &K) -> Option<&mut H>;

    fn get_or_insert_with(&mut self, key: &K, insert_fn: impl Fn() -> H) -> &mut H;

    fn summarize<'a: 'b, 'b>(&'a self) -> Option<H>
    where
        &'a Self: IntoIterator<Item = (&'b K, &'b H)>,
        H: 'b,
        Self: 'a + 'b,
        K: 'b,
    {
        let seed = self.into_iter().next()?.1.empty_clone();
        Some(self.into_iter().fold(seed, |mut agg, item| {
            agg.merge_borrowed(item.1);
            agg
        }))
    }
}

pub trait Summarize<H> {
    fn summarize(self) -> Option<H>;
}

impl<'b, H, Set> Summarize<H> for Set
where
    H: 'b + HistogramSetItem,
    Set: Iterator<Item = &'b H>,
{
    fn summarize(self) -> Option<H> {
        let mut peekable = self.peekable();
        let seed = peekable.peek()?.empty_clone();
        Some(peekable.fold(seed, |mut agg, item| {
            agg.merge_borrowed(item);
            agg
        }))
    }
}

pub trait BaseHistogram<T, C>: HistogramSetItem {
    /// Type of a bin in this histogram
    type Bin;

    /// Instantiate a histogram with the given number of maximum bins
    fn new(n_bins: usize) -> Self;

    /// Insert a new data point into this histogram
    fn insert(&mut self, value: T, count: C);

    /// Count the total number of data points in this histogram (over all bins)
    fn count(&self) -> C;
}

pub trait Median<T> {
    /// Estimate the median value of the data points in this histogram
    fn median(&self) -> Option<T>;
}

pub trait ContinuousValue:
    Float + ExchangeData + Sum + AbsDiffEq + RelativeEq + UlpsEq + Debug
{
}
impl<T: Float + ExchangeData + Sum + AbsDiffEq + RelativeEq + UlpsEq + Debug> ContinuousValue for T {}

pub trait DiscreteValue: Ord + Eq + Hash + Copy + ExchangeData + Debug {}
impl<T: Ord + Eq + Hash + Copy + ExchangeData + Debug> DiscreteValue for T {}

pub trait HistogramSetItem: Clone {
    type Serializable: From<Self> + Into<Self> + ExchangeData;

    /// Merge another instance of this type into this histogram
    fn merge(&mut self, other: Self);

    /// Merge another instance of this type into this histogram
    fn merge_borrowed(&mut self, other: &Self);

    /// Return an empty clone of the item that has otherwise identical attributes (e.g. number of maximum bins)
    fn empty_clone(&self) -> Self;
}

pub trait FromData<Tr, D> {
    fn from_data(tree: &Tr, data: &[D], bins: usize) -> Self;
}

pub trait FindNodeLabel<T> {
    fn find_node_label(&self, node: &NodeIndex) -> Option<T>;
}
