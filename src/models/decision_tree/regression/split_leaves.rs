use std::fmt::Debug;
use data::serialization::Serializable;
use models::decision_tree::histogram_generics::{ContinuousValue, DiscreteValue};
use models::decision_tree::operators::SplitLeaves;
use models::decision_tree::regression::histogram::{TargetValueHistogramSet, FindSplits, FindNodeLabel};
use models::decision_tree::split_improvement::SplitImprovement;
use models::decision_tree::tree::DecisionTree;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::*;
use timely::dataflow::{Scope, Stream};
use timely::progress::nested::product::Product;
use timely::progress::Timestamp;
use timely::Data;

impl<S, Ts1, T, L, I> SplitLeaves<T, L, S, I>
    for Stream<S, (DecisionTree<T, L>, <TargetValueHistogramSet<T, L> as Serializable>::Serializable)>
where
    (DecisionTree<T, L>, <TargetValueHistogramSet<T, L> as Serializable>::Serializable): Data,
    S: Scope<Timestamp = Product<Ts1, u64>>,
    Ts1: Timestamp,
    T: DiscreteValue + Debug,
    L: ContinuousValue + Debug,
    I: SplitImprovement<T, L, HistogramData = TargetValueHistogramSet<T, L>>,
{
    fn split_leaves(&self, levels: u64, bins: u64) -> Stream<S, (usize, DecisionTree<T, L>)> {
        self.unary(Pipeline, "BuildTree", |_, _| {
            move |input, output| {
                input.for_each(move |time, data| {
                    for (mut tree, flat_histograms) in data.drain(..) {
                        let histograms: TargetValueHistogramSet<_, _> = Serializable::from_serializable(flat_histograms);
                        let current_iteration = time.inner;
                        let mut split_leaves = 0;
                        debug!("Begin splitting phase");
                        if current_iteration < levels {
                            let splits = histograms.find_best_splits(&tree.unlabeled_leaves());
                            for (node, rule) in splits {
                                // TODO: add intermediary labels to nodes
                                tree.split(node, rule, None);
                            }
                        } else {
                            for leaf in tree.unlabeled_leaves() {
                                if let Some(label) = histograms.find_node_label(&leaf) {
                                    debug!("Labeling node {:?} with {:?}", leaf, label);
                                    tree.label(leaf, label);
                                }
                            }
                        }
                        output.session(&time).give((split_leaves, tree));
                    }
                });
            }
        });
        unimplemented!()
    }
}
