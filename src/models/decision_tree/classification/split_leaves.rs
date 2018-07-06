use models::decision_tree::classification::histogram::FeatureValueHistogramSet;
use models::decision_tree::histogram_generics::*;
use models::decision_tree::operators::SplitLeaves;
use models::decision_tree::split_improvement::SplitImprovement;
use models::decision_tree::tree::{DecisionTree, Rule};
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::Operator;
use timely::dataflow::{Scope, Stream};
use timely::progress::nested::product::Product;
use timely::progress::Timestamp;

impl<
        S: Scope<Timestamp = Product<Ts1, u64>>,
        Ts1: Timestamp,
        T: ContinuousValue,
        L: DiscreteValue,
        I: Clone + SplitImprovement<T, L, HistogramData = FeatureValueHistogramSet<T, L>> + 'static,
    > SplitLeaves<T, L, S, I>
    for Stream<
        S,
        (
            DecisionTree<T, L>,
            <FeatureValueHistogramSet<T, L> as HistogramSetItem>::Serializable,
        ),
    >
{
    fn split_leaves(
        &self,
        levels: u64,
        improvement_algo: I,
    ) -> Stream<S, (usize, DecisionTree<T, L>)> {
        self.unary(Pipeline, "BuildTree", |_, _| {
            move |input, output| {
                let improvement_algo = improvement_algo.clone();
                input.for_each(move |time, data| {
                    for (mut tree, histograms) in data.drain(..) {
                        let mut histograms: FeatureValueHistogramSet<_, _> = histograms.into();

                        let current_iteration = time.inner;
                        let mut split_leaves = 0;
                        debug!("Begin splitting phase");
                        if current_iteration < levels {
                            tree.unlabeled_leaves()
                                .iter()
                                // ignores leaves where no data points arrive
                                .filter_map(|leaf| Some((leaf, histograms.get(&leaf)?)))
                                .for_each(|(leaf, node_histograms)| {
                                    let (split_attr, (delta, split_location)) = node_histograms
                                        .iter()
                                        .map(|(attr, attr_histograms)| {
                                            // merge all histograms for a node & attribute, combining the ones
                                            // for individual labels
                                            let merged_histograms = attr_histograms.summarize().expect("Summarize attribute histograms");

                                            // calculate impurity delta for each candidate split and return the highest
                                            let best_delta_and_split = merged_histograms
                                                .candidate_splits()
                                                .iter()
                                                .map(|candidate_split| {
                                                    let delta = improvement_algo
                                                        .split_improvement(
                                                            &histograms,
                                                            *leaf,
                                                            attr,
                                                            *candidate_split,
                                                        )
                                                        .unwrap();
                                                    trace!("Calculating candidate split for attr {:?} at {:?}; delta = {:?}", attr, candidate_split, delta);
                                                    (delta, *candidate_split)
                                                })
                                                .max_by(|a, b| {
                                                    a.0
                                                        .partial_cmp(&b.0)
                                                        .unwrap_or(::std::cmp::Ordering::Less)
                                                })
                                                .expect("Choose maximum split delta");
                                            debug!(
                                                "Best split for attribute {:?}: {:?} with delta {:?}",
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

                                    if delta > T::zero() {
                                        debug!("Splitting tree node {:?} with attribute {:?} < {:?}: delta {:?}",leaf, split_attr, split_location, delta);
                                        let label = histograms.find_node_label(leaf);

                                        tree.split(
                                            *leaf,
                                            Rule::threshold(split_attr, split_location),
                                            label,
                                        );
                                        split_leaves += 1;
                                    } else {
                                        let label = histograms
                                            .find_node_label(leaf)
                                            .expect("Get node label");
                                        debug!("Splitting tree node {:?} would result in a negative delta; labeling node with {:?}", leaf, label);
                                        tree.label(*leaf, label);
                                    }
                                });
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
        })
    }
}
