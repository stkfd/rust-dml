/*use data::serialization::Serializable;
use data::TrainingData;
use fnv::FnvHashMap;
use models::decision_tree::histogram_generics::{
    BaseHistogram, ContinuousValue, DiscreteValue, FromData, HistogramSet, HistogramSetItem,
};
use models::decision_tree::operators::*;
use models::decision_tree::tree::{DecisionTree, Node};
use std::fmt::Debug;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::{Exchange, Operator};
use timely::dataflow::{Scope, Stream};
use timely::progress::nested::product::Product;
use timely::progress::Timestamp;
use timely::ExchangeData;


impl<S, T, L, K, H> AggregateHistograms<S, T, L>
    for Stream<
        S,
        (
            DecisionTree<T, L>,
            <HistogramSet<K, H> as Serializable>::Serializable,
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
            <HistogramSet<K, H> as Serializable>::Serializable,
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
*/
