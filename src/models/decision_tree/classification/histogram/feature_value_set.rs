use models::decision_tree::histogram_generics::*;
use models::decision_tree::histogram_generics::{
    SerializableFnvHistogramSet, SerializableVecHistogramSet,
};
use models::decision_tree::tree::NodeIndex;
use super::*;

type K = NodeIndex;
type Inner<T, L> = VecHistogramSet<FnvHistogramSet<L, Histogram<T>>>;

/// Nested set of histograms that contains
/// Node -> Attribute Index -> Target Value -> Histogram with feature values
#[derive(Clone)]
pub struct FeatureValueHistogramSet<T: ContinuousValue, L: DiscreteValue>(
    FnvHistogramSet<NodeIndex, VecHistogramSet<FnvHistogramSet<L, Histogram<T>>>>,
);

#[derive(Clone, Abomonation)]
pub struct SerializableFeatureValueHistogramSet<T: ContinuousValue, L: DiscreteValue>(
    SerializableFnvHistogramSet<NodeIndex, SerializableVecHistogramSet<SerializableFnvHistogramSet<L, Histogram<T>>>>,
);

impl<T: ContinuousValue, L: DiscreteValue> Default for FeatureValueHistogramSet<T, L> {
    fn default() -> Self {
        FeatureValueHistogramSet(Default::default())
    }
}

impl<T: ContinuousValue, L: DiscreteValue>
    HistogramSet<K, Inner<T, L>> for FeatureValueHistogramSet<T, L>
{
    fn get(&self, key: &K) -> Option<&Inner<T, L>> { self.0.get(key) }

    fn get_mut(&mut self, key: &K) -> Option<&mut Inner<T, L>> { self.0.get_mut(key) }

    fn get_or_insert_with(&mut self, key: &K, insert_fn: impl Fn() -> Inner<T, L>) -> &mut Inner<T, L> { self.0.get_or_insert_with(key, insert_fn) }
}

impl<T: ContinuousValue, L: DiscreteValue> From<FeatureValueHistogramSet<T, L>> for SerializableFeatureValueHistogramSet<T, L> {
    fn from(set: FeatureValueHistogramSet<T, L>) -> Self {
        SerializableFeatureValueHistogramSet(set.0.into())
    }
}

impl<T: ContinuousValue, L: DiscreteValue> Into<FeatureValueHistogramSet<T, L>> for SerializableFeatureValueHistogramSet<T, L> {
    fn into(self) -> FeatureValueHistogramSet<T, L> {
        FeatureValueHistogramSet(self.0.into())
    }
}

impl<T: ContinuousValue, L: DiscreteValue> HistogramSetItem for FeatureValueHistogramSet<T, L> {
    type Serializable = SerializableFeatureValueHistogramSet<T, L>;

    fn merge(&mut self, other: Self) { self.0.merge(other.0) }

    fn merge_borrowed(&mut self, other: &Self) { self.0.merge_borrowed(&other.0) }

    fn empty_clone(&self) -> Self {
        Self::default()
    }
}
