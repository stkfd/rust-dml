use super::*;
use models::spdt::histogram::operators::TreeWithHistograms;
use models::spdt::histogram::*;
use models::spdt::tree::DecisionTree;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::Operator;
use timely::dataflow::{Scope, Stream};
use timely::progress::nested::product::Product;
use timely::progress::Timestamp;
use timely::Data;

/// Operator that splits the unlabeled leaf nodes in a decision tree according to Histogram data
pub trait SplitLeaves<T, L, S: Scope> {
    /// Split all unlabeled leaves where a split would improve the classification accuracy
    fn split_leaves<I: Impurity>(&self, levels: u64, bins: u64) -> Stream<S, (usize, DecisionTree<T, L>)>;
}

impl<S: Scope<Timestamp = Product<Ts1, u64>>, Ts1: Timestamp, L: Data + Copy + PartialEq + Debug>
    SplitLeaves<f64, L, S> for Stream<S, TreeWithHistograms<f64, L>>
{
    fn split_leaves<I: Impurity>(&self, levels: u64, bins: u64) -> Stream<S, (usize, DecisionTree<f64, L>)> {
        self.unary(Pipeline, "BuildTree", |_| {
            move |input, output| {
                input.for_each(move |time, data| {
                    for (mut tree, histograms, n_attributes) in data.drain(..) {
                        let current_iteration = time.inner;
                        let mut split_leaves = 0;
                        debug!("Begin splitting phase");
                        debug!("Attributes: {}", n_attributes);
                        if current_iteration < levels {
                            for leaf in tree.unlabeled_leaves() {
                                if histograms.node_has_samples(leaf) {
                                    let (split_attr, (delta, split_location)) = (0..n_attributes)
                                        .map(|attr| {
                                            let merged_histograms = histograms
                                                .get_by_node_attribute(leaf, attr)
                                                .expect("Get histograms by node/attribute")
                                                .iter()
                                                .fold(Histogram::new(bins as usize), move |mut acc, h| {
                                                    acc.merge(&h.1);
                                                    acc
                                                });

                                            // calculate impurity delta for each candidate split and return the highest
                                            let best_delta_and_split = merged_histograms
                                                .candidate_splits()
                                                .iter()
                                                .map(|candidate_split| {
                                                    let delta =
                                                        I::impurity_delta(
                                                            &histograms,
                                                            leaf,
                                                            attr,
                                                            *candidate_split,
                                                        ).unwrap();
                                                    trace!("Calculating candidate split for attr {} at {}; delta = {}", attr, candidate_split, delta);
                                                    (delta, *candidate_split)
                                                })
                                                .max_by(|a, b| {
                                                    a.0
                                                        .partial_cmp(&b.0)
                                                        .unwrap_or(::std::cmp::Ordering::Less)
                                                })
                                                .expect("Choose maximum split delta");
                                            debug!(
                                                "Best split for attribute {}: {} with delta {}",
                                                attr,
                                                best_delta_and_split.1,
                                                best_delta_and_split.0
                                            );
                                            (attr, best_delta_and_split)
                                        })
                                        .max_by(|(_, (delta1, _)), (_, (delta2, _))| {
                                            delta1
                                                .partial_cmp(&delta2)
                                                .unwrap_or(::std::cmp::Ordering::Less)
                                        })
                                        .expect("Choose best split attribute");

                                    if delta > 0. {
                                        debug!("Splitting tree node {:?} with attribute {} < {}: delta {}",leaf, split_attr, split_location, delta);
                                        let label = histograms.get_node_label(leaf);

                                        tree.split(
                                            leaf,
                                            Rule::new(split_attr, split_location),
                                            label,
                                        );
                                        split_leaves += 1;
                                    } else {
                                        let label = histograms
                                            .get_node_label(leaf)
                                            .expect("Get node label");
                                        debug!("Splitting tree node {:?} would result in a negative delta; labeling node with {:?}", leaf, label);
                                        tree.label(leaf, label);
                                    }
                                } else {
                                    debug!("No samples reach node {:?}", leaf);
                                }
                            }
                        } else {
                            for leaf in tree.unlabeled_leaves() {
                                if let Some(label) = histograms.get_node_label(leaf) {
                                    debug!("Labeling node {:?} with {:?}", leaf, label);
                                    tree.label(leaf, label);
                                }
                            }
                        }
                        output.session(&time).give((split_leaves, tree));
                    }
                });
            }
        })
    }
}