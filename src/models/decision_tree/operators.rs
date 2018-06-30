
use timely::ExchangeData;
use models::decision_tree::tree::DecisionTree;
use models::decision_tree::split_improvement::SplitImprovement;
use data::TrainingData;
use timely::dataflow::Stream;
use timely::Data;
use timely::dataflow::Scope;

/// Extension trait for timely `Stream`
pub trait CreateHistograms<S: Scope, T: Data, L: Data, O: Data> {
    /// Takes a set of `TrainingData` and a stream of decision trees (as they are created).
    /// For each decision tree, compiles a set of histograms describing the samples which
    /// arrive at each unlabeled leaf node in the tree.
    fn create_histograms(
        &self,
        training_data: &Stream<S, TrainingData<T, L>>,
        bins: usize,
        data_cache_size: usize,
    ) -> Stream<S, O>;
}

pub trait AggregateHistograms<S: Scope, T: ExchangeData, L: ExchangeData> {
    fn aggregate_histograms(&self) -> Self;
}

/// Operator that splits the unlabeled leaf nodes in a decision tree according to Histogram data
pub trait SplitLeaves<T, L, S: Scope, I: SplitImprovement<T, L>> {
    /// Split all unlabeled leaves where a split would improve the classification accuracy
    /// if the innermost `time` exceeds the given maximum `levels`, the operator will stop
    /// splitting and instead only label the remaining leaf nodes with the most commonly
    /// occuring labels that reach the node.
    fn split_leaves(&self, levels: u64, bins: u64) -> Stream<S, (usize, DecisionTree<T, L>)>;
}
