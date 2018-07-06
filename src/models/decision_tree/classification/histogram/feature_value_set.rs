use models::decision_tree::histogram_generics::*;
use models::decision_tree::histogram_generics::{
    SerializableBTreeHistogramSet, SerializableVecHistogramSet,
};
use models::decision_tree::tree::NodeIndex;
use super::*;

type K = NodeIndex;
type Inner<T, L> = VecHistogramSet<BTreeHistogramSet<L, Histogram<T>>>;

/// Nested set of histograms that contains
/// Node -> Attribute Index -> Target Value -> Histogram with feature values
#[derive(Clone)]
pub struct FeatureValueHistogramSet<T: ContinuousValue, L: DiscreteValue>(
    BTreeHistogramSet<NodeIndex, VecHistogramSet<BTreeHistogramSet<L, Histogram<T>>>>,
);

#[derive(Clone, Abomonation)]
pub struct SerializableFeatureValueHistogramSet<T: ContinuousValue, L: DiscreteValue>(
    SerializableBTreeHistogramSet<NodeIndex, SerializableVecHistogramSet<SerializableBTreeHistogramSet<L, Histogram<T>>>>,
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

    fn select<'a>(&mut self, keys: impl IntoIterator<Item = &'a K>, callback: impl Fn(&mut Inner<T, L>))
    where
        K: 'a { self.0.select(keys, callback) }

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

/*
/// Holds a set of histograms describing the samples reaching
/// the leaf nodes in a decision tree.
#[allow(type_complexity)]
#[derive(Clone, Debug, Abomonation)]
pub struct HistogramCollection<T: Float, L> {
    collection: Vec<(NodeIndex, Vec<Vec<(L, Histogram<T>)>>)>,
}

impl<T: Float, L> Default for HistogramCollection<T, L> {
    fn default() -> Self {
        HistogramCollection { collection: vec![] }
    }
}

impl<T, L> HistogramCollection<T, L>
where
    T: ContinuousValue,
    L: Copy + PartialEq,
{
    /// Get a mutable reference to a histogram for a node/attribute/label
    /// combination
    pub fn get_mut(
        &mut self,
        node_index: NodeIndex,
        attribute: usize,
        label: L,
    ) -> Option<&mut Histogram<T>> {
        self.collection
            .iter_mut()
            .find(|(i, _)| *i == node_index)
            .and_then(|n| n.1.get_mut(attribute))
            .and_then(|by_label| {
                by_label
                    .iter_mut()
                    .find(|(l, _)| *l == label)
                    .and_then(|h| Some(&mut h.1))
            })
    }

    // Determine if any samples arrive at the given node
    pub fn node_has_samples(&self, node_index: NodeIndex) -> bool {
        self.collection.iter().any(|(i, _)| *i == node_index)
    }

    /// Get the histogram with the given node/attribute/label combination
    pub fn get(&self, node_index: NodeIndex, attribute: usize, label: L) -> Option<&Histogram<T>> {
        self.get_by_node_attribute(node_index, attribute)
            .and_then(|by_label| {
                by_label
                    .iter()
                    .find(|(l, _)| *l == label)
                    .and_then(|h| Some(&h.1))
            })
    }

    /// Gets all histograms at a given node/attribute combination
    #[inline]
    pub fn get_by_node_attribute(
        &self,
        node_index: NodeIndex,
        attribute: usize,
    ) -> Option<&Vec<(L, Histogram<T>)>> {
        self.collection
            .iter()
            .find(|(i, _)| *i == node_index)
            .and_then(|n| n.1.get(attribute))
    }

    /// Gets all histograms at a given node
    #[inline]
    pub fn get_by_node(&self, node_index: NodeIndex) -> Option<&Vec<Vec<(L, Histogram<T>)>>> {
        self.collection
            .iter()
            .find(|(i, _)| *i == node_index)
            .and_then(|(_, histograms)| Some(histograms))
    }

    /// Insert a histogram into this collection. Automatically merges
    /// the histogram into another histogram that describes the same
    /// node/attribute/label combination, if such a histogram already
    /// exists.
    pub fn insert(
        &mut self,
        histogram: Histogram<T>,
        node_index: NodeIndex,
        attribute_index: usize,
        label: L,
    ) {
        let by_node =
            if let Some(position) = self.collection.iter().position(|(i, _)| *i == node_index) {
                &mut self.collection[position].1
            } else {
                self.collection.push((node_index, vec![]));
                let p = self.collection.len() - 1;
                &mut self.collection[p].1
            };

        if by_node.len() <= attribute_index {
            by_node.resize(attribute_index + 1, vec![]);
        };
        let by_attr = &mut by_node[attribute_index];

        by_attr.push((label, histogram));
    }

    /// Merge another collection of Histograms into this collection
    pub fn merge(&mut self, mut other: HistogramCollection<T, L>) {
        for (node_index, mut by_node) in other.collection.drain(..) {
            for (attr, mut by_attr) in by_node.drain(..).enumerate() {
                for (label, new_histogram) in by_attr.drain(..) {
                    if let Some(histogram) = self.get_mut(node_index, attr, label) {
                        histogram.merge(&new_histogram);
                    }
                    if self.get(node_index, attr, label).is_none() {
                        self.insert(new_histogram, node_index, attr, label);
                    }
                }
            }
        }
    }

    /// Get the most frequently occuring label at the given node
    pub fn get_node_label(&self, node: NodeIndex) -> Option<L> {
        // FIXME: should merge all histograms for the node first
        let histograms = &self.get_by_node(node)?.get(0)?;

        histograms
            .iter()
            .map(|(label, h)| (label, h.sum_total()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Less))
            .and_then(|most_common| Some(*most_common.0))
    }
}
*/
