use super::*;
use fnv::FnvHashMap;
use models::spdt::tree::{DecisionTree, Node, NodeIndex};
use models::spdt::TrainingData;
use std::hash::Hash;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::{aggregation::Aggregate, Map, Operator};
use timely::dataflow::{Scope, Stream};
use timely::progress::nested::product::Product;
use timely::progress::Timestamp;
use timely::{Data, ExchangeData};

pub type TreeWithHistograms<T, L> = (DecisionTree<T, L>, HistogramCollection<L>, usize);

#[derive(Clone, Copy, Debug, Abomonation, PartialEq, Eq, Hash)]
pub struct HistogramIndex {
    pub node_index: usize,
    pub attribute: usize,
    pub class_index: usize,
}

#[allow(type_complexity)]
#[derive(Clone, Debug, Abomonation)]
pub struct HistogramCollection<L> {
    collection: Vec<(NodeIndex, Vec<Vec<(L, Histogram)>>)>,
}

impl<L> Default for HistogramCollection<L> {
    fn default() -> Self {
        HistogramCollection { collection: vec![] }
    }
}

impl<L: Copy + PartialEq> HistogramCollection<L> {
    pub fn get_mut(
        &mut self,
        node_index: NodeIndex,
        attribute: usize,
        label: L,
    ) -> Option<&mut Histogram> {
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

    // determine if any samples arrive at the node
    pub fn node_has_samples(&self, node_index: NodeIndex) -> bool {
        self.collection.iter().any(|(i, _)| *i == node_index)
    }

    pub fn get(&self, node_index: NodeIndex, attribute: usize, label: L) -> Option<&Histogram> {
        self.get_by_node_attribute(node_index, attribute)
            .and_then(|by_label| {
                by_label
                    .iter()
                    .find(|(l, _)| *l == label)
                    .and_then(|h| Some(&h.1))
            })
    }

    #[inline]
    pub fn get_by_node_attribute(
        &self,
        node_index: NodeIndex,
        attribute: usize,
    ) -> Option<&Vec<(L, Histogram)>> {
        self.collection
            .iter()
            .find(|(i, _)| *i == node_index)
            .and_then(|n| n.1.get(attribute))
    }

    #[inline]
    pub fn get_by_node(&self, node_index: NodeIndex) -> Option<&Vec<Vec<(L, Histogram)>>> {
        self.collection
            .iter()
            .find(|(i, _)| *i == node_index)
            .and_then(|(_, histograms)| Some(histograms))
    }

    pub fn insert(
        &mut self,
        histogram: Histogram,
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
    pub fn merge(&mut self, mut other: HistogramCollection<L>) {
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

    pub fn get_node_label(&self, node: NodeIndex) -> Option<L> {
        let histograms = &self.get_by_node(node)?.get(0)?;

        histograms
            .iter()
            .map(|(label, h)| (label, h.sum_total()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Less))
            .and_then(|most_common| Some(*most_common.0))
    }
}

pub trait CreateHistograms<S: Scope, T: Data, L: Data> {
    fn create_histograms(
        &self,
        training_data: &Stream<S, TrainingData<T, L>>,
        bins: usize,
        data_cache_size: usize,
    ) -> Stream<S, TreeWithHistograms<T, L>>;
}

impl<
        S: Scope<Timestamp = Product<TsOuter, u64>>,
        TsOuter: Timestamp,
        T: Data + PartialOrd + Into<f64> + Copy,
        L: Data + Into<usize> + Copy + PartialEq,
    > CreateHistograms<S, T, L> for Stream<S, DecisionTree<T, L>>
{
    fn create_histograms(
        &self,
        training_data: &Stream<S, TrainingData<T, L>>,
        bins: usize,
        data_cache_size: usize,
    ) -> Stream<S, TreeWithHistograms<T, L>> {
        let worker = self.scope().index();
        self.binary_frontier(
            training_data,
            Pipeline,
            Pipeline,
            "CreateHistograms",
            |_| {
                let mut data_stash = FnvHashMap::default();
                let mut tree_stash = FnvHashMap::default();
                let mut n_attributes = None;

                move |in_tree, in_data, out| {
                    in_data.for_each(|time, data| {
                        debug!(
                            "Worker {} received training data at {:?}",
                            worker,
                            time.time()
                        );

                        let (cached_samples, current_index, sample_cache) = data_stash
                            .entry(time.time().outer.clone())
                            .or_insert_with(|| (0_usize, 0_usize, Vec::new()));

                        for datum in data.drain(..) {
                            let cols = datum.x().cols();
                            if *current_index >= sample_cache.len() {
                                if *cached_samples < data_cache_size {
                                    sample_cache.push(datum);
                                    *current_index += 1;
                                } else {
                                    *cached_samples -= sample_cache[0].x().cols();
                                    sample_cache[0] = datum;
                                    *current_index = 1;
                                }
                            } else {
                                *cached_samples -= sample_cache[*current_index].x().cols();
                                sample_cache[*current_index] = datum;
                                *current_index += 1;
                            }
                            *cached_samples += cols;
                        }
                    });

                    in_tree.for_each(|time, trees| {
                        debug!(
                            "Worker {} received temporary tree at {:?}",
                            worker,
                            time.time()
                        );
                        if tree_stash.contains_key(&time) || trees.len() > 1 {
                            panic!("Received more than one tree for a time")
                        }
                        let tree = trees.drain(..).next().unwrap();
                        tree_stash.insert(time, tree);
                    });

                    let frontiers = [in_data.frontier(), in_tree.frontier()];
                    for (time, tree) in &tree_stash {
                        // received the decision tree for this time
                        if frontiers.iter().all(|f| !f.less_equal(&time)) {
                            debug!("Worker {} collecting histograms", worker);
                            debug!("{:?}", data_stash.keys());
                            let (_, _, data) = data_stash.get(&time.outer).expect("Retrieve data");
                            //let tree = tree_stash.remove(time).expect("Retrieve decision tree");
                            let mut histograms = HistogramCollection::<L>::default();

                            for training_data in data {
                                let x = training_data.x();
                                let y = training_data.y();

                                // make sure every data chunk has the same number of columns
                                if let Some(n_attributes) = n_attributes {
                                    assert_eq!(n_attributes, x.cols());
                                } else {
                                    n_attributes = Some(x.cols());
                                }

                                for (x_row, y_i) in x.outer_iter().zip(y.iter()) {
                                    let node_index = tree.descend_iter(x_row)
                                        .last()
                                        .expect("Navigate to leaf node");
                                    if let Node::Leaf { label: None } = tree[node_index] {
                                        for (i_attr, x_i) in x_row.iter().enumerate() {
                                            if histograms.get(node_index, i_attr, *y_i).is_none() {
                                                histograms.insert(
                                                    Histogram::new(bins),
                                                    node_index,
                                                    i_attr,
                                                    *y_i,
                                                );
                                            }
                                            let mut histogram = histograms
                                                .get_mut(node_index, i_attr, *y_i)
                                                .unwrap();

                                            histogram.update((*x_i).into());
                                        }
                                    }
                                }
                            }

                            out.session(&time).give((
                                tree.clone(),
                                histograms,
                                n_attributes.expect("Unwrap attribute count"),
                            ));
                        }
                    }

                    tree_stash.retain(|time, _| !frontiers.iter().all(|f| !f.less_equal(time)));
                    data_stash.retain(|time, _| {
                        !frontiers
                            .iter()
                            .all(|f| !f.less_equal(&Product::new(time.clone(), <u64>::max_value())))
                    });
                }
            },
        )
    }
}

pub trait AggregateHistograms<S: Scope, T: ExchangeData, L: ExchangeData> {
    fn aggregate_histograms(&self) -> Stream<S, TreeWithHistograms<T, L>>;
}

impl<S: Scope, T: ExchangeData, L: ExchangeData + Hash + Eq + Copy> AggregateHistograms<S, T, L>
    for Stream<S, TreeWithHistograms<T, L>>
{
    fn aggregate_histograms(&self) -> Stream<S, TreeWithHistograms<T, L>> {
        self.map(|x| (0, x))
            .aggregate::<_, TreeWithHistograms<T, L>, _, _, _>(
                |_key, (tree, histograms, n_attr), agg| {
                    agg.0 = tree;
                    agg.1.merge(histograms);
                    agg.2 = n_attr;
                },
                |_, histograms| histograms,
                |_| 0_u64,
            )
    }
}
