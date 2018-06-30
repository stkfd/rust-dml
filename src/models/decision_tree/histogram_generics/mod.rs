use data::serialization::Serializable;
use num_traits::Float;
use std::hash::Hash;
use timely::Data;

mod btree;
mod fnv;

pub use self::btree::BTreeHistogramSet;
pub use self::fnv::FnvHistogramSet;

pub trait HistogramSet<K, T: HistogramSetItem>: Default {
    fn get(&self, key: &K) -> Option<&T>;
    fn get_mut(&mut self, key: &K) -> Option<&mut T>;

    fn get_or_insert_with(&mut self, key: &K, insert_fn: impl Fn() -> T) -> &mut T;

    fn select<'a>(
        &mut self,
        keys: impl IntoIterator<Item = &'a K>,
        callback: impl Fn(&mut T),
    ) where
        K: 'a;

    fn summarize<'a: 'b, 'b>(&'a self) -> Option<T>
    where
        &'a Self: IntoIterator<Item = (&'b K, &'b T)>,
        T: 'b,
        Self: 'a + 'b,
        K: 'b,
    {
        self.into_iter().map(|(_k, h)| h).summarize()
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

pub trait BaseHistogram<T>: HistogramSetItem {
    /// Type of a bin in this histogram
    type Bin;

    /// Instantiate a histogram with the given number of maximum bins
    fn new(n_bins: usize) -> Self;

    /// Insert a new data point into this histogram
    fn insert(&mut self, value: T);

    /// Count the total number of data points in this histogram (over all bins)
    fn count(&self) -> u64;

    /// Estimate the median value of the data points in this histogram
    fn median(&self) -> T;
}

pub trait ContinuousValue: Float + Data {}
impl<T: Float + Data> ContinuousValue for T {}

pub trait DiscreteValue: Ord + Eq + Hash + Copy + Data {}
impl<T: Ord + Eq + Hash + Copy + Data> DiscreteValue for T {}

pub trait HistogramSetItem: Clone + Serializable {
    /// Merge another instance of this type into this histogram
    fn merge(&mut self, other: Self);

    /// Merge another instance of this type into this histogram
    fn merge_borrowed(&mut self, other: &Self);

    /// Return an empty clone of the item that has otherwise identical attributes (e.g. number of maximum bins)
    fn empty_clone(&self) -> Self;
}
