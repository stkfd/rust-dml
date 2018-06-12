use super::*;
use models::decision_tree::tree::NodeIndex;

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
    T: HFloat,
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
        let histograms = &self.get_by_node(node)?.get(0)?;

        histograms
            .iter()
            .map(|(label, h)| (label, h.sum_total()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Less))
            .and_then(|most_common| Some(*most_common.0))
    }
}
