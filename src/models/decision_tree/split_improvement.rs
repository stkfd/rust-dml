//! A trait for measurements of how good a split of a node with a specific condition would be

use models::decision_tree::tree::NodeIndex;

pub trait SplitImprovement<T, L> {
    type HistogramData;

    /// Calculates how much the impurity in the tree would be reduced if
    /// it was split at the given node, attribute and split point. The `HistogramCollection`
    /// is expected to contain all histograms for each attribute and label at this node.
    fn split_improvement(
        &self,
        histogram_data: &Self::HistogramData,
        node: NodeIndex,
        attribute: usize,
        split_at: T,
    ) -> Option<T>
    where
        L: Copy + PartialEq + ::std::fmt::Debug;
}
