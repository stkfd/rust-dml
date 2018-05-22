#![allow(unknown_lints)]

mod histogram;
mod impurity;
mod tree;

use self::histogram::operators::*;
use self::histogram::Histogram;
use self::impurity::*;
use self::tree::*;
use data::dataflow::ExchangeEvenly;
use data::serialization::*;
use fnv::FnvHashMap;
use models::StreamingSupModel;
use ndarray::prelude::*;
use std::hash::Hash;
use std::marker::PhantomData;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::*;
use timely::dataflow::{Scope, Stream};
use timely::progress::nested::product::Product;
use timely::progress::Timestamp;
use timely::Data;
use timely::ExchangeData;
use Result;

pub struct StreamingDecisionTree<I: Impurity> {
    levels: u64,
    points_per_worker: u64,
    bins: usize,
    impurity_algo: PhantomData<I>,
}

impl<I: Impurity> StreamingDecisionTree<I> {
    pub fn new(levels: u64, points_per_worker: u64, bins: usize) -> Self {
        StreamingDecisionTree {
            levels,
            points_per_worker,
            bins,
            impurity_algo: PhantomData,
        }
    }
}

impl<L, I: Impurity> StreamingSupModel<TrainingData<f64, L>, AbomonableArray2<f64>, (), ()>
    for StreamingDecisionTree<I>
where
    L: ExchangeData + Into<usize> + Copy + Eq + Hash + 'static,
{
    /// Predict output from inputs.
    fn predict<S: Scope>(
        &mut self,
        _scope: &mut S,
        _inputs: Stream<S, AbomonableArray2<f64>>,
    ) -> Result<Stream<S, ()>> {
        unimplemented!()
    }

    /// Train the model using inputs and targets.
    fn train<S: Scope>(
        &mut self,
        scope: &mut S,
        data: Stream<S, TrainingData<f64, L>>,
    ) -> Result<Stream<S, ()>> {
        scope.scoped::<u64, _, _>(|data_segments_scope| {
            let training_data = data.enter(data_segments_scope)
                .segment_training_data(data_segments_scope.peers() as u64 * self.points_per_worker)
                .exchange_evenly();

            data_segments_scope.scoped::<u64, _, _>(|tree_iter_scope| {
                let bins = self.bins;
                let init_tree = if tree_iter_scope.index() == 0 {
                    vec![DecisionTree::<f64, L>::default()]
                } else {
                    vec![]
                };

                let worker = tree_iter_scope.index();
                let (loop_handle, cycle) = tree_iter_scope.loop_variable(self.levels, 1);
                init_tree
                    .to_stream(tree_iter_scope)
                    .concat(&cycle)
                    .broadcast()
                    .create_histograms(&training_data.enter(tree_iter_scope), self.bins)
                    .aggregate_histograms()
                    .map(move |(mut tree, histograms, n_attributes)| {
                        for leaf in tree.unlabeled_leaves() {
                            let (split_attr, (_delta, split_location)) = (0..n_attributes)
                                .map(|attr| {
                                    let merged_histograms = histograms
                                        .get_by_node_attribute(leaf, attr)
                                        .expect("Getting histogram by node and attribute")
                                        .iter()
                                        .fold(Histogram::new(bins), move |mut acc, h| {
                                            acc.merge(&h.1);
                                            acc
                                        });

                                    // calculate impurity delta for each candidate split and return the highest
                                    let best_delta_and_split = merged_histograms
                                        .uniform(bins)
                                        .iter()
                                        .map(|candidate_split| {
                                            let delta =
                                                I::impurity_delta(
                                                    &histograms,
                                                    leaf,
                                                    attr,
                                                    *candidate_split,
                                                ).unwrap();
                                            (delta, *candidate_split)
                                        })
                                        .max_by(|a, b| {
                                            a.0
                                                .partial_cmp(&b.0)
                                                .unwrap_or(::std::cmp::Ordering::Less)
                                        })
                                        .expect("Choose maximum split delta");
                                    (attr, best_delta_and_split)
                                })
                                .max_by(|(_, (delta1, _)), (_, (delta2, _))| {
                                    delta1
                                        .partial_cmp(&delta2)
                                        .unwrap_or(::std::cmp::Ordering::Less)
                                })
                                .expect("Choose best split attribute");
                            
                            tree.split(leaf, Rule::new(split_attr, split_location));
                        }

                        tree
                    })
                    .inspect(move |_x| println!("{}", worker))
                    .connect_loop(loop_handle);
            });
        });
        Ok(vec![()].to_stream(scope))
    }
}

#[derive(Clone, Abomonation, Debug)]
pub struct TrainingData<T, L> {
    pub x: AbomonableArray2<T>,
    pub y: AbomonableArray1<L>,
}

impl<T, L> TrainingData<T, L> {
    pub fn x<'a, 'b: 'a>(&'b self) -> ArrayView2<'a, T> {
        self.x.view()
    }
    pub fn x_mut<'a, 'b: 'a>(&'b mut self) -> ArrayViewMut2<'a, T> {
        self.x.view_mut()
    }
    pub fn y<'a, 'b: 'a>(&'b self) -> ArrayView1<'a, L> {
        self.y.view()
    }
    pub fn y_mut<'a, 'b: 'a>(&'b mut self) -> ArrayViewMut1<'a, L> {
        self.y.view_mut()
    }
}

trait SegmentTrainingData<S: Scope, T: Data, L: Data> {
    fn segment_training_data(&self, items_per_segment: u64) -> Stream<S, TrainingData<T, L>>;
}

impl<S: Scope<Timestamp = Product<Ts, u64>>, Ts: Timestamp, T: Data, L: Data>
    SegmentTrainingData<S, T, L> for Stream<S, TrainingData<T, L>>
{
    fn segment_training_data(&self, items_per_segment: u64) -> Stream<S, TrainingData<T, L>> {
        self.unary_frontier(Pipeline, "SegmentTrainingData", |_| {
            let mut stash: FnvHashMap<_, (Product<_, u64>, u64)> = FnvHashMap::default();

            move |input, output| {
                input.for_each(|cap, data| {
                    let time = stash
                        .entry(cap.clone())
                        .or_insert_with(|| (cap.time().clone(), 0));
                    for training_data in data.iter() {
                        time.1 += training_data.x().rows() as u64;
                        if time.1 > items_per_segment {
                            time.0.inner += 1;
                        }
                    }
                    output.session(&cap.delayed(&time.0)).give_content(data);
                });

                stash.retain(|time, _| !input.frontier().less_equal(time.time()));
            }
        })
    }
}
