use models::decision_tree::histogram_generics::*;
use models::decision_tree::tree::NodeIndex;
use super::*;

type K = NodeIndex;
type Inner<T, L> = VecHistogramSet<FnvHistogramSet<T, Histogram<L>>>;

/// Nested set of histograms that contains
/// Node -> Attribute Index -> Feature Value -> Histogram with target values
#[derive(Clone)]
pub struct TargetValueHistogramSet<T: DiscreteValue, L: ContinuousValue>(
    BTreeHistogramSet<NodeIndex, VecHistogramSet<FnvHistogramSet<T, Histogram<L>>>>,
);

#[derive(Clone, Abomonation)]
pub struct SerializableTargetValueHistogramSet<T: DiscreteValue, L: ContinuousValue>(
    SerializableBTreeHistogramSet<NodeIndex, SerializableVecHistogramSet<SerializableFnvHistogramSet<T, SerializableHistogram<L>>>>,
);

impl<T: DiscreteValue, L: ContinuousValue> Default for TargetValueHistogramSet<T, L> {
    fn default() -> Self {
        TargetValueHistogramSet(Default::default())
    }
}

impl<T: DiscreteValue, L: ContinuousValue>
    HistogramSet<K, Inner<T, L>> for TargetValueHistogramSet<T, L>
{
    fn get(&self, key: &K) -> Option<&Inner<T, L>> { self.0.get(key) }

    fn select<'a>(&mut self, keys: impl IntoIterator<Item = &'a K>, callback: impl Fn(&mut Inner<T, L>))
    where
        K: 'a { self.0.select(keys, callback) }

    fn get_mut(&mut self, key: &K) -> Option<&mut Inner<T, L>> { self.0.get_mut(key) }

    fn get_or_insert_with(&mut self, key: &K, insert_fn: impl Fn() -> Inner<T, L>) -> &mut Inner<T, L> { self.0.get_or_insert_with(key, insert_fn) }
}

impl<T: DiscreteValue, L: ContinuousValue> From<TargetValueHistogramSet<T, L>> for SerializableTargetValueHistogramSet<T, L> {
    fn from(set: TargetValueHistogramSet<T, L>) -> Self {
        SerializableTargetValueHistogramSet(set.0.into())
    }
}

impl<T: DiscreteValue, L: ContinuousValue> Into<TargetValueHistogramSet<T, L>> for SerializableTargetValueHistogramSet<T, L> {
    fn into(self) -> TargetValueHistogramSet<T, L> {
        TargetValueHistogramSet(self.0.into())
    }
}

impl<T: DiscreteValue, L: ContinuousValue> HistogramSetItem for TargetValueHistogramSet<T, L> {
    type Serializable = SerializableTargetValueHistogramSet<T, L>;

    fn merge(&mut self, other: Self) { self.0.merge(other.0) }

    fn merge_borrowed(&mut self, other: &Self) { self.0.merge_borrowed(&other.0) }

    fn empty_clone(&self) -> Self {
        Self::default()
    }
}
