use data::providers::IntSliceIndex;

use super::*;

pub(crate) trait AssignPoints<S: Scope, D: Data + ::std::fmt::Debug> {
    fn assign_points(
        &self,
        points_stream: &Stream<S, (IntSliceIndex<usize>, AbomonableArray2<D>)>,
    ) -> Stream<S, AbomonableAggregationStatistics<D>>;
}

impl<S: Scope<Timestamp = Product<T, usize>>, T: Timestamp, D> AssignPoints<S, D>
    for Stream<S, AbomonableArray2<D>>
where
    D: ::std::fmt::Debug + Data + NumAssignOps + Scalar + FromPrimitive + ScalarOperand,
{
    fn assign_points(
        &self,
        points_stream: &Stream<S, (IntSliceIndex<usize>, AbomonableArray2<D>)>,
    ) -> Stream<S, AbomonableAggregationStatistics<D>> {
        let worker_index = self.scope().index();
        self.binary_frontier(
            &points_stream,
            Pipeline,
            Pipeline,
            "AssignPoints",
            |_, _| {
                let mut point_stash = Vec::new();
                let mut centroid_stash = HashMap::new();

                move |in_centroids, in_points, out| {
                    in_centroids.for_each(|time, data| {
                        debug!(
                            "Worker {} receiving {} sets of centroids",
                            worker_index,
                            data.len()
                        );
                        let entry = centroid_stash.entry(time.retain()).or_insert_with(Vec::new);
                        entry.extend(data.drain(..));
                    });

                    in_points.for_each(|time, data| {
                        debug!(
                            "Worker {} receiving {} data chunks",
                            worker_index,
                            data.len()
                        );
                        assert_eq!(time.inner, 0);
                        point_stash.extend(data.drain(..));
                    });

                    let frontiers = [in_centroids.frontier(), in_points.frontier()];
                    for (cap, centroid_list) in &mut centroid_stash {
                        // if neither input can produce data at `time`, compute statistics
                        if frontiers.iter().all(|f| !f.less_equal(cap.time())) {
                            debug!("Worker {} processing centroid/point data", worker_index);
                            let mut session = out.session(&cap);

                            for centroids in centroid_list.drain(..) {
                                let centroids_view = centroids.view();
                                let mut agg = AggregationStatistics::new(
                                    centroids_view.rows(),
                                    centroids_view.cols(),
                                );

                                for &(slice_index, ref points) in &point_stash {
                                    let points_view: ArrayView<
                                        _,
                                        _,
                                    > = points.into();
                                    agg.collect_assignment_statistics(&points_view, &centroids_view, &slice_index);
                                }

                                session.give(agg.into());
                            }
                        }
                    }

                    centroid_stash.retain(|_time, list| !list.is_empty());
                }
            },
        )
    }
}
