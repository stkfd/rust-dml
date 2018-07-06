use data::TrainingData;
use fnv::FnvHashMap;
use models::decision_tree::histogram_generics::*;
use models::decision_tree::tree::DecisionTree;
use std::fmt::Debug;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::Capability;
use timely::dataflow::operators::{Exchange, Operator};
use timely::dataflow::{Scope, Stream};
use timely::progress::{nested::product::Product, Timestamp};
use timely::Data;
use timely::ExchangeData;

/// Extension trait for timely `Stream`
pub trait CreateHistograms<S: Scope, T: Data, L: Data> {
    /// Takes a set of `TrainingData` and a stream of decision trees (as they are created).
    /// For each decision tree, compiles a set of histograms describing the samples which
    /// arrive at each unlabeled leaf node in the tree.
    fn create_histograms<H: HistogramSetItem + FromData<DecisionTree<T, L>, TrainingData<T, L>>>(
        &self,
        training_data: &Stream<S, TrainingData<T, L>>,
        bins: usize,
        data_cache_size: usize,
    ) -> Stream<S, (DecisionTree<T, L>, H::Serializable)>;
}

impl<S, TsOuter, T, L> CreateHistograms<S, T, L> for Stream<S, DecisionTree<T, L>>
where
    S: Scope<Timestamp = Product<TsOuter, u64>>,
    TsOuter: Timestamp,
    T: Data,
    L: Data,
{
    fn create_histograms<H: HistogramSetItem + FromData<DecisionTree<T, L>, TrainingData<T, L>>>(
        &self,
        training_data: &Stream<S, TrainingData<T, L>>,
        bins: usize,
        data_cache_size: usize,
    ) -> Stream<S, (DecisionTree<T, L>, H::Serializable)> {
        let worker = self.scope().index();
        self.binary_frontier(
            training_data,
            Pipeline,
            Pipeline,
            "CreateHistograms",
            |_, _| {
                let mut data_stash = FnvHashMap::default();
                let mut tree_stash = FnvHashMap::default();

                move |in_tree, in_data, out| {
                    in_data.for_each(|cap, data| {
                        let outer_time = cap.time().outer.clone();

                        debug!(
                            "Worker {} received training data at {:?}",
                            worker, outer_time
                        );

                        let (cached_sample_count, current_index, sample_cache) = data_stash
                            .entry(outer_time)
                            .or_insert_with(|| (0_usize, 0_usize, Vec::new()));

                        for datum in data.drain(..) {
                            let cols = datum.x().cols();
                            if *current_index >= sample_cache.len() {
                                if *cached_sample_count < data_cache_size {
                                    sample_cache.push(datum);
                                    *current_index += 1;
                                } else {
                                    *cached_sample_count -= sample_cache[0].x().cols();
                                    sample_cache[0] = datum;
                                    *current_index = 1;
                                }
                            } else {
                                *cached_sample_count -= sample_cache[*current_index].x().cols();
                                sample_cache[*current_index] = datum;
                                *current_index += 1;
                            }
                            *cached_sample_count += cols;
                        }
                    });

                    in_tree.for_each(|time, trees| {
                        let time = time.retain();
                        debug!(
                            "Worker {} received temporary tree at {:?}",
                            worker,
                            time.time()
                        );
                        if tree_stash.contains_key(&time) || trees.len() > 1 {
                            panic!("Received more than one tree for a time")
                        }
                        let tree = trees.drain(..).next().unwrap();
                        tree_stash.insert(time, Some(tree));
                    });

                    let frontiers = [in_data.frontier(), in_tree.frontier()];
                    for (time, tree_opt) in &mut tree_stash {
                        // received the decision tree for this time
                        if frontiers.iter().all(|f| !f.less_equal(&time)) {
                            let tree = tree_opt.take().unwrap();
                            debug!("Worker {} collecting histograms", worker);
                            let (_, _, data) = data_stash.get(&time.outer).expect("Retrieve data");

                            let histograms = H::from_data(&tree, data, bins);

                            out.session(&time).give((tree.clone(), histograms.into()));
                        }
                    }

                    data_stash.retain(|time, _| {
                        !frontiers
                            .iter()
                            .all(|f| !f.less_equal(&Product::new(time.clone(), <u64>::max_value())))
                    });
                    tree_stash.retain(|_time, tree_opt| tree_opt.is_some());
                }
            },
        )
    }
}

/// Operator that splits the unlabeled leaf nodes in a decision tree according to Histogram data
pub trait SplitLeaves<T, L, S: Scope, I> {
    /// Split all unlabeled leaves where a split would improve the classification accuracy
    /// if the innermost `time` exceeds the given maximum `levels`, the operator will stop
    /// splitting and instead only label the remaining leaf nodes with the most commonly
    /// occuring labels that reach the node.
    fn split_leaves(
        &self,
        levels: u64,
        improvement_algo: I,
    ) -> Stream<S, (usize, DecisionTree<T, L>)>;
}

pub trait AggregateHistograms<S: Scope, SetS: ExchangeData> {
    fn aggregate_histograms<Set>(&self) -> Self
    where
        Set: HistogramSetItem<Serializable = SetS> + 'static,
        SetS: From<Set> + Into<Set>;
}

impl<S, Tree, SetS> AggregateHistograms<S, SetS> for Stream<S, (Tree, SetS)>
where
    S: Scope,
    SetS: ExchangeData,
    Tree: ExchangeData + PartialEq + Debug,
{
    fn aggregate_histograms<Set>(&self) -> Self
    where
        Set: HistogramSetItem<Serializable = SetS> + 'static,
        SetS: From<Set> + Into<Set>,
    {
        let worker = self.scope().index();
        self.exchange(|_| 0_u64)
            .unary_frontier(Pipeline, "MergeHistogramSets", |_, _| {
                let mut stash: FnvHashMap<Capability<_>, Option<(Tree, Set)>> =
                    FnvHashMap::default();
                move |input, output| {
                    // receive and merge incoming histograms
                    input.for_each(|cap_ref, data| {
                        debug!("Receiving histograms for merging");
                        let opt_entry = stash.entry(cap_ref.retain()).or_insert_with(|| {
                            let (tree, f_hist) = data.pop().expect("First tree/histogram set");
                            Some((tree, f_hist.into()))
                        });

                        let (tree, histogram_set) = opt_entry.as_mut().unwrap();
                        for (new_tree, new_histogram_set) in data.drain(..) {
                            // make sure all trees in this timestamp are the same
                            assert_eq!(*tree, new_tree);
                            histogram_set.merge(new_histogram_set.into());
                        }
                    });

                    for (cap, stash_entry) in &mut stash {
                        // send out merged histograms at the end of each timestamp
                        if !input.frontier().less_equal(cap) {
                            debug!(
                                "Sending merged histograms for timestamp {:?} on worker {}",
                                cap, worker
                            );
                            let (tree, merged_set) = stash_entry.take().unwrap();
                            output.session(cap).give((tree.clone(), merged_set.into()));
                        }
                    }

                    stash.retain(|_, entry| entry.is_some());
                }
            })
    }
}
