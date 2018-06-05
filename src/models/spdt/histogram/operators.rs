use super::*;
use super::collection::HistogramCollection;
use fnv::FnvHashMap;
use models::spdt::tree::{DecisionTree, Node};
use models::spdt::TrainingData;
use std::hash::Hash;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::{aggregation::Aggregate, Map, Operator};
use timely::dataflow::{Scope, Stream};
use timely::progress::nested::product::Product;
use timely::progress::Timestamp;
use timely::{Data, ExchangeData};

/// Convenience type for a combination of:
/// (decision tree, histogram set for the tree, number of attributes in the dataset)
/// 
/// TODO: the number of attributes does not really fit here semantically, however the
/// `CreateHistogram` operator which returns this type is the best place to determine
/// this number.
pub type TreeWithHistograms<T, L> = (DecisionTree<T, L>, HistogramCollection<L>, usize);

/// Extension trait for timely `Stream`
pub trait CreateHistograms<S: Scope, T: Data, L: Data> {
    /// Takes a set of `TrainingData` and a stream of decision trees (as they are created).
    /// For each decision tree, compiles a set of histograms describing the samples which
    /// arrive at each unlabeled leaf node in the tree.
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
