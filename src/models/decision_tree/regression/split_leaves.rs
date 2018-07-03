use data::serialization::Serializable;
use models::decision_tree::histogram_generics::{ContinuousValue, DiscreteValue};
use models::decision_tree::operators::SplitLeaves;
use models::decision_tree::regression::histogram::loss_functions::WeightedLoss;
use models::decision_tree::regression::histogram::{
    FindNodeLabel, FindSplits, TargetValueHistogramSet,
};
use models::decision_tree::tree::DecisionTree;
use std::fmt::Debug;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::*;
use timely::dataflow::{Scope, Stream};
use timely::progress::nested::product::Product;
use timely::progress::Timestamp;
use timely::Data;

impl<S, Ts1, T, L, I> SplitLeaves<T, L, S, I>
    for Stream<
        S,
        (
            DecisionTree<T, L>,
            <TargetValueHistogramSet<T, L> as Serializable>::Serializable,
        ),
    > where
    (
        DecisionTree<T, L>,
        <TargetValueHistogramSet<T, L> as Serializable>::Serializable,
    ): Data,
    S: Scope<Timestamp = Product<Ts1, u64>>,
    Ts1: Timestamp,
    T: DiscreteValue + Debug,
    L: ContinuousValue + Debug,
    I: Clone + WeightedLoss<L> + 'static,
{
    fn split_leaves(
        &self,
        levels: u64,
        loss_func: I,
        bins: u64,
    ) -> Stream<S, (usize, DecisionTree<T, L>)> {
        self.unary(Pipeline, "BuildTree", |_, _| {
            move |input, output| {
                let loss_func = loss_func.clone();
                input.for_each(|time, data| {
                    for (mut tree, flat_histograms) in data.drain(..) {
                        let histograms: TargetValueHistogramSet<_, _> =
                            Serializable::from_serializable(flat_histograms);
                        let current_iteration = time.inner;
                        let mut split_leaves = 0;
                        if current_iteration < levels {
                            debug!("Begin splitting phase");
                            let splits = histograms.find_best_splits(&tree.unlabeled_leaves(), &loss_func);
                            for (node, rule) in splits {
                                // TODO: add intermediary labels to nodes
                                tree.split(node, rule, histograms.find_node_label(&node));
                            }
                        } else {
                            debug!("Labeling remaining leaf nodes");
                            for leaf in tree.unlabeled_leaves() {
                                histograms.find_node_label(&leaf).map(|label| {
                                    debug!("Labeling node {:?} with {:?}", leaf, label);
                                    tree.label(leaf, label);
                                });
                            }
                        }
                        output.session(&time).give((split_leaves, tree));
                    }
                });
            }
        })
    }
}
