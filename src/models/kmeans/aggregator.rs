use data::providers::IndexesSlice;
use data::serialization::*;
use ndarray::prelude::*;
use ndarray::ScalarOperand;
use ndarray::Zip;
use ndarray_linalg::types::Scalar;
use ndarray_linalg::Norm;
use num_traits::cast::FromPrimitive;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{AddAssign, DivAssign};
use timely::dataflow::{channels::pact::Pipeline, operators::Unary, Scope, Stream};
use timely::Data;

#[derive(Clone, Debug)]
pub(crate) struct AggregationStatistics<T: Debug> {
    pub centroid_assignments: Vec<(usize, usize)>,
    pub cluster_sums: Array2<T>,
    pub cluster_counts: Array1<usize>,
}

#[derive(Abomonation, Clone)]
pub(crate) struct AbomonableAggregationStatistics<T: Debug> {
    pub centroid_assignments: Vec<(usize, usize)>,
    pub cluster_sums: AbomonableArray2<T>,
    pub cluster_counts: AbomonableArray1<usize>,
}

impl<T: Data + Debug> From<AbomonableAggregationStatistics<T>> for AggregationStatistics<T> {
    fn from(from: AbomonableAggregationStatistics<T>) -> Self {
        AggregationStatistics {
            centroid_assignments: from.centroid_assignments,
            cluster_sums: from.cluster_sums.into(),
            cluster_counts: from.cluster_counts.into(),
        }
    }
}

impl<T: Data + Debug> From<AggregationStatistics<T>> for AbomonableAggregationStatistics<T> {
    fn from(from: AggregationStatistics<T>) -> AbomonableAggregationStatistics<T> {
        AbomonableAggregationStatistics {
            centroid_assignments: from.centroid_assignments,
            cluster_sums: from.cluster_sums.into(),
            cluster_counts: from.cluster_counts.into(),
        }
    }
}

impl<T> AggregationStatistics<T>
where
    T: Scalar + FromPrimitive + AddAssign<T> + ScalarOperand + DivAssign<T>,
{
    pub fn new(centroids: usize, cols: usize) -> Self {
        AggregationStatistics {
            centroid_assignments: Vec::new(),
            cluster_sums: Array2::zeros((centroids, cols)),
            cluster_counts: Array1::zeros(centroids),
        }
    }

    /// Assigns the given points to the given set of centroids, sums up the values of the assigned
    /// points and counts how many were assigned to each centroid.
    pub fn consume_points<'a>(
        &mut self,
        points: &ArrayView2<'a, T>,
        centroids: &ArrayView2<'a, T>,
        slice_index: &impl IndexesSlice<Idx = usize>,
    ) where
        T: 'a,
    {
        for (point_idx, point) in points.outer_iter().enumerate() {
            // find closest centroid
            let centroid_idx = centroids
                .outer_iter()
                .enumerate()
                .map(|(index, candidate_centroid)| {
                    let distance = (&point - &candidate_centroid).norm_l2();
                    (index, distance)
                })
                .min_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap_or(::std::cmp::Ordering::Less))
                .unwrap()
                .0;

            // save assignment
            self.centroid_assignments
                .push((slice_index.absolute_index(point_idx), centroid_idx));
            // add point to sum of all points assigned to centroid
            self.cluster_sums
                .subview_mut(Axis(0), centroid_idx)
                .add_assign(&point);
            // increment count of assigned points
            self.cluster_counts[centroid_idx] += 1;
        }
    }

    /// Estimate a new set of centroids from the assignment statistics in this struct
    pub fn centroid_estimate(&self) -> Array2<T> {
        let sums: ArrayView2<_> = (&self.cluster_sums).into();
        let counts: ArrayView1<_> = (&self.cluster_counts).into();
        let mut estimates: Array2<_> = sums.to_owned();

        Zip::from(estimates.outer_iter_mut())
            .and(&counts)
            .apply(|mut sum, &count| {
                if count > 0 {
                    sum /= T::from_usize(count).unwrap()
                } else {
                    sum.fill(T::from_usize(0).unwrap())
                }
            });

        estimates
    }
}

impl<'a, T> AddAssign<&'a AggregationStatistics<T>> for &'a mut AggregationStatistics<T>
where
    T: Scalar + AddAssign<T>,
{
    /// Add the cluster aggregation statistics of another instance to this one
    fn add_assign(&mut self, rhs: &AggregationStatistics<T>) {
        self.centroid_assignments
            .extend_from_slice(rhs.centroid_assignments.as_slice());

        let mut sums: ArrayViewMut2<_> = (&mut self.cluster_sums).into();
        let other_sums: ArrayView2<_> = (&rhs.cluster_sums).into();
        sums += &other_sums;

        let mut counts: ArrayViewMut1<_> = (&mut self.cluster_counts).into();
        let other_counts: ArrayView1<_> = (&rhs.cluster_counts).into();
        counts += &other_counts;
    }
}

pub(crate) trait AccumulateStatistics<G: Scope, T: Data + Debug> {
    /// Accumulates all incoming `AggregationStatistic` instances into one per timestamp by summing
    /// them up.
    fn accumulate_statistics(
        &self,
        default: AggregationStatistics<T>,
    ) -> Stream<G, AbomonableAggregationStatistics<T>>;
}

impl<S: Scope, D: Scalar + AddAssign<D> + Data + Debug> AccumulateStatistics<S, D>
    for Stream<S, AbomonableAggregationStatistics<D>>
{
    fn accumulate_statistics(
        &self,
        default: AggregationStatistics<D>,
    ) -> Stream<S, AbomonableAggregationStatistics<D>> {
        let mut accums = HashMap::new();
        self.unary_notify(
            Pipeline,
            "Accumulate",
            vec![],
            move |input, output, notificator| {
                input.for_each(|time, data| {
                    let mut agg = accums
                        .entry(time.time().clone())
                        .or_insert_with(|| default.clone());
                    for incoming in data.drain(..) {
                        agg += &<AggregationStatistics<D>>::from(incoming);
                    }
                    notificator.notify_at(time);
                });

                notificator.for_each(|time, _, _| {
                    if let Some(accum) = accums.remove(&time) {
                        output.session(&time).give(accum.into());
                    }
                });
            },
        )
    }
}
