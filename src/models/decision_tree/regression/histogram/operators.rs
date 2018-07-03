use data::serialization::Serializable;
use data::TrainingData;
use fnv::FnvHashMap;
use models::decision_tree::histogram_generics::{
    BaseHistogram, ContinuousValue, DiscreteValue, HistogramSet, HistogramSetItem,
};
use models::decision_tree::operators::*;
use models::decision_tree::regression::histogram::TargetValueHistogramSet;
use models::decision_tree::tree::{DecisionTree, Node};
use std::fmt::Debug;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::{Exchange, Operator};
use timely::dataflow::{Scope, Stream};
use timely::progress::nested::product::Product;
use timely::progress::Timestamp;
use timely::ExchangeData;

impl<S, TsOuter, T, L>
    CreateHistograms<
        S,
        T,
        L,
        (
            DecisionTree<T, L>,
            <TargetValueHistogramSet<T, L> as Serializable>::Serializable,
        ),
    > for Stream<S, DecisionTree<T, L>>
where
    S: Scope<Timestamp = Product<TsOuter, u64>>,
    TsOuter: Timestamp,
    T: DiscreteValue + Debug,
    L: ContinuousValue + Debug,
{
    fn create_histograms(
        &self,
        training_data: &Stream<S, TrainingData<T, L>>,
        bins: usize,
        data_cache_size: usize,
    ) -> Stream<
        S,
        (
            DecisionTree<T, L>,
            <TargetValueHistogramSet<T, L> as Serializable>::Serializable,
        ),
    > {
        let worker = self.scope().index();
        self.binary_frontier(
            training_data,
            Pipeline,
            Pipeline,
            "CreateHistograms",
            |_, _| {
                let mut data_stash = FnvHashMap::default();
                let mut tree_stash = FnvHashMap::default();
                let mut n_attributes = None;

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

                            let mut histograms = TargetValueHistogramSet::<T, L>::default();

                            for training_data in data {
                                let x = training_data.x();
                                let y = training_data.y();

                                // make sure every data chunk has the same number of columns
                                if let Some(n_attributes) = n_attributes {
                                    assert_eq!(n_attributes, x.cols());
                                } else {
                                    // initialize histogram set once the number of columns is known
                                    n_attributes = Some(x.cols());
                                }

                                for (x_row, y_i) in x.outer_iter().zip(y.iter()) {
                                    let node_index = tree
                                        .descend_iter(x_row)
                                        .last()
                                        .expect("Navigate to leaf node");
                                    if let Node::Leaf { label: None } = tree[node_index] {
                                        let node_histograms = histograms
                                            .get_or_insert_with(&node_index, Default::default);
                                        for (i_attr, x_i) in x_row.iter().enumerate() {
                                            node_histograms
                                                .get_or_insert_with(&i_attr, Default::default)
                                                .get_or_insert_with(x_i, || {
                                                    BaseHistogram::new(bins)
                                                })
                                                .insert(*y_i);
                                        }
                                    }
                                }
                            }

                            out.session(&time)
                                .give((tree.clone(), histograms.into_serializable()));
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

impl<S, T, L> AggregateHistograms<S, T, L>
    for Stream<
        S,
        (
            DecisionTree<T, L>,
            <TargetValueHistogramSet<T, L> as Serializable>::Serializable,
        ),
    > where
    S: Scope,
    T: DiscreteValue + ExchangeData + Debug,
    L: ContinuousValue + ExchangeData + Debug,
{
    fn aggregate_histograms(
        &self,
    ) -> Stream<
        S,
        (
            DecisionTree<T, L>,
            <TargetValueHistogramSet<T, L> as Serializable>::Serializable,
        ),
    > {
        let worker = self.scope().index();
        self.exchange(|_| 0_u64)
            .unary_frontier(Pipeline, "MergeHistogramSets", |_, _| {
                let mut stash: FnvHashMap<
                    _,
                    Option<(DecisionTree<T, L>, TargetValueHistogramSet<T, L>)>,
                > = FnvHashMap::default();
                move |input, output| {
                    // receive and merge incoming histograms
                    input.for_each(|time, data| {
                        debug!("Receiving histograms for merging");
                        let opt_entry = stash.entry(time.retain()).or_insert_with(|| {
                            let (tree, f_hist) = data.pop().expect("First tree/histogram set");
                            Some((tree, Serializable::from_serializable(f_hist)))
                        });

                        opt_entry.iter_mut().for_each(move |(tree, histogram_set)| {
                            for (new_tree, new_histogram_set) in data.drain(..) {
                                // make sure all trees in this timestamp are the same
                                assert_eq!(*tree, new_tree);
                                histogram_set
                                    .merge(Serializable::from_serializable(new_histogram_set));
                            }
                        });
                    });

                    for (time, stash_entry) in &mut stash {
                        // send out merged histograms at the end of each timestamp
                        if !input.frontier().less_equal(time) {
                            debug!(
                                "Sending merged histograms for timestamp {:?} on worker {}",
                                time.time(),
                                worker
                            );
                            let (tree, merged_set) = stash_entry.take().unwrap();
                            output
                                .session(&time)
                                .give((tree.clone(), merged_set.into_serializable()));
                        }
                    }

                    stash.retain(|_, entry| entry.is_some());
                }
            })
    }
}
/*
#[derive(Abomonation, Clone)]
pub struct FlattenedHistogramSet<T, L: ContinuousValue> {
    pub histograms: Vec<(
        NodeIndex,
        usize,
        T,
        <Histogram<L> as Serializable>::Serializable,
    )>,
    pub attributes: usize,
    pub bins: usize,
}

impl<T, L> From<HistogramSet<T, L>> for FlattenedHistogramSet<T, L>
where
    T: PartialOrd + Hash + Eq + Copy,
    L: ContinuousValue,
{
    fn from(histogram_set: HistogramSet<T, L>) -> Self {
        let attributes = histogram_set.attributes;
        let bins = histogram_set.bins;
        let histograms = histogram_set
            .into_vec()
            .into_iter()
            .map(|(node, attr, x, hist)| (node, attr, x, hist.into_serializable()))
            .collect();
        FlattenedHistogramSet {
            histograms,
            attributes,
            bins,
        }
    }
}

impl<T, L> From<FlattenedHistogramSet<T, L>> for HistogramSet<T, L>
where
    T: PartialOrd + Hash + Eq + Copy,
    L: ContinuousValue,
{
    fn from(mut f_histogram_set: FlattenedHistogramSet<T, L>) -> Self {
        let mut histograms = HistogramSet::new(f_histogram_set.attributes, f_histogram_set.bins);

        for (node, attr, x, histogram) in f_histogram_set.histograms.drain(..) {
            histograms.merge_histogram(node, attr, x, Serializable::from_serializable(histogram));
        }

        histograms
    }
}
*/
