//! The K-Means Clustering Algorithm.

use self::aggregator::*;
use data::dataflow::AsyncResult;
use data::providers::IntSliceIndex;
use data::providers::{DataSource, DataSourceSpec};
use data::serialization::*;
use models::UnSupModel;
use ndarray::indices;
use ndarray::prelude::*;
use ndarray::{NdProducer, ScalarOperand, Zip};
use ndarray_linalg::{Norm, Scalar};
use num_traits::cast::FromPrimitive;
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::{AddAssign, DivAssign, Sub};
use std::sync::mpsc;
use timely::dataflow::channels::pact::Pipeline;
use timely::progress::Timestamp;
use timely::{dataflow::{operators::*, Scope, Stream},
             progress::nested::product::Product,
             Data,
             ExchangeData};
use Result;

pub use self::initializers::KMeansInitializer;

mod aggregator;
mod convergence;
pub mod initializers;

pub use self::convergence::*;

const CHUNK_SIZE: usize = 5000;

pub struct Kmeans<Item: Data, Init: KMeansInitializer<Item>> {
    n_clusters: usize,
    cols: usize,
    centroids: AsyncResult<AbomonableArray2<Item>>,
    end_criteria: ConvergenceCriteria<Item>,
    phantom_data: PhantomData<Init>,
}

impl<Item: Data, Init: KMeansInitializer<Item>> Kmeans<Item, Init> {
    pub fn new(n_clusters: usize, cols: usize, end_criteria: ConvergenceCriteria<Item>) -> Self {
        Kmeans {
            n_clusters,
            cols,
            centroids: AsyncResult::Uninitialized,
            end_criteria,
            phantom_data: PhantomData,
        }
    }

    pub fn centroids(&mut self) -> Option<ArrayView2<Item>> {
        self.centroids.get().ok().map(|a| a.view())
    }
}

impl<Item, Init> UnSupModel<AbomonableArray2<Item>, AbomonableArray2<usize>, AbomonableArray2<Item>>
    for Kmeans<Item, Init>
where
    Item: ExchangeData
        + FromPrimitive
        + Scalar
        + Float
        + Display
        + ScalarOperand
        + AddAssign<Item>
        + DivAssign<Item>
        + Sub<Item>,
    Init: KMeansInitializer<Item>,
{
    fn predict<S: Scope, Sp: DataSourceSpec<AbomonableArray2<Item>>>(
        &mut self,
        scope: &mut S,
        inputs: Sp,
    ) -> Result<Stream<S, AbomonableArray2<usize>>> {
        if scope.index() != 0 { return Ok(vec!().to_stream(scope)); }
        let centroids = self.centroids.get()?.view();

        let mut provider = inputs.to_provider()?;
        let points: Array2<_> = provider.all()?.into();

        let mut assignments = unsafe { Array2::<usize>::uninitialized((points.rows(), 2)) };
        Zip::from(assignments.genrows_mut())
            .and(points.genrows())
            .and(indices(points.genrows().raw_dim()))
            .apply(|mut assignment, point, point_idx| {
                let centroid_index = centroids
                    .outer_iter()
                    .map(|centroid| (&point - &centroid).norm_l2())
                    .enumerate()
                    .min_by(|&(_, a), &(_, b)| {
                        a.partial_cmp(&b).unwrap_or(::std::cmp::Ordering::Less)
                    })
                    .unwrap()
                    .0;
                assignment[0] = point_idx;
                assignment[1] = centroid_index;
            });

        Ok(vec![AbomonableArray::from(assignments)].to_stream(scope))

        // TODO: finish distributing prediction work across different workers
        /*let worker_index = scope.index();
        let points_stream = provider
            .chunk_indices(CHUNK_SIZE)
            .unwrap()
            .enumerate()
            .to_stream(scope)
            .exchange(|&(chunk_num, _)| chunk_num as u64)
            .map(move |(_, slice_index)| {
                debug!("{:?}", slice_index);
                let mut provider = inputs.clone().to_provider().unwrap();
                (
                    slice_index,
                    AbomonableArray::from(provider.slice(slice_index).unwrap()),
                )
            })
            .map(|(slice_index, chunk_a)| {
                let chunk: Array2<_> = chunk_a.into();
                let assignments = Array2::<usize>::uninitialized((chunk.rows(), 2));
                chunk
                    .outer_iter()
                    .enumerate()
                    .map(|(i, point)| {
                        let centroid_index = centroids.outer_iter()
                            .map(move |centroid| {
                                (&point - &centroid).norm_l2()
                            })
                            .enumerate()
                            .min_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap_or(::std::cmp::Ordering::Less))
                            .unwrap()
                            .0;
                        (slice_index.absolute_index(i), centroid_index)
                    })
            });*/    }

    fn train<S: Scope, D: DataSourceSpec<AbomonableArray2<Item>>>(
        &mut self,
        scope: &mut S,
        input_spec: D,
    ) -> Result<Stream<S, AbomonableArray2<Item>>> {
        let n_clusters = self.n_clusters;
        let cols = self.cols;

        let end_criteria = self.end_criteria.clone();
        let max_iterations = self.end_criteria
            .max_iterations
            .unwrap_or(<usize>::max_value());

        let mut provider = input_spec.to_provider()?;
        let initial_centroids: Vec<AbomonableArray2<Item>> =
            vec![Init::select_initial_centroids(&mut provider, n_clusters, cols)?];

        debug!(
            "Selected initial centroids: {}",
            initial_centroids[0].view()
        );

        let (result_sender, result_receiver) = mpsc::channel();
        if scope.index() == 0 { self.centroids = AsyncResult::Receiver(result_receiver); }

        let results = scope.scoped(|inner| {
            let worker_index = inner.index();
            debug!("Constructing worker {}", inner.index());

            let points_stream = provider
                .chunk_indices(CHUNK_SIZE)
                .unwrap()
                .enumerate()
                .to_stream(inner)
                .exchange(|&(chunk_num, _)| chunk_num as u64)
                .map(move |(_, slice_index)| {
                    debug!("{:?}", slice_index);
                    let mut provider = input_spec.clone().to_provider().unwrap();
                    (
                        slice_index,
                        provider.slice(slice_index).unwrap(),
                    )
                });

            let (loop_handle, loop_stream) = inner.loop_variable(max_iterations, 1);

            let (done, next_iteration) = initial_centroids
                .to_stream(inner)
                .filter(move |_| worker_index == 0)
                .concat(&loop_stream)
                // checks whether the convergence criteria are met and aborts the loop
                .end_condition(end_criteria);

            next_iteration
                .broadcast()
                .assign_points(&points_stream)
                .exchange(|_| 0u64)
                .accumulate_statistics(AggregationStatistics::new(n_clusters, cols))
                .map(move |stats| {
                    debug!(
                        "Worker {}: Aggregated all assignments, calculating new centroids",
                        worker_index
                    );
                    AggregationStatistics::from(stats)
                        .centroid_estimate()
                        .into()
                })
                .connect_loop(loop_handle);

            done.inspect(move |c| {
                debug!("worker {}", worker_index);
                debug!("Finished: {}", c.view());
                result_sender
                    .send(c.clone())
                    .expect("Extracting result centroids");
            }).leave()
        });

        Ok(results)
    }
}

trait AssignPoints<S: Scope, D: Data + ::std::fmt::Debug> {
    fn assign_points(
        &self,
        points_stream: &Stream<S, (IntSliceIndex<usize>, AbomonableArray2<D>)>,
    ) -> Stream<S, AbomonableAggregationStatistics<D>>;
}

impl<S: Scope<Timestamp = Product<T, usize>>, T: Timestamp, D> AssignPoints<S, D>
    for Stream<S, AbomonableArray2<D>>
where
    D: ::std::fmt::Debug
        + Data
        + AddAssign<D>
        + DivAssign<D>
        + Scalar
        + FromPrimitive
        + ScalarOperand,
{
    fn assign_points(
        &self,
        points_stream: &Stream<S, (IntSliceIndex<usize>, AbomonableArray2<D>)>,
    ) -> Stream<S, AbomonableAggregationStatistics<D>> {
        let worker_index = self.scope().index();
        self.binary_frontier(&points_stream, Pipeline, Pipeline, "AssignPoints", |_| {
            let mut point_stash = Vec::new();
            let mut centroid_stash = HashMap::new();

            move |in_centroids, in_points, out| {
                in_centroids.for_each(|time, data| {
                    debug!(
                        "Worker {} receiving {} sets of centroids",
                        worker_index,
                        data.len()
                    );
                    let entry = centroid_stash.entry(time.clone()).or_insert_with(Vec::new);
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
                                let points_view: ArrayView<_, _> = points.into();
                                agg.consume_points(&points_view, &centroids_view, &slice_index);
                            }

                            session.give(agg.into());
                        }
                    }
                }

                centroid_stash.retain(|_time, list| !list.is_empty());
            }
        })
    }
}

trait EndCondition<S: Scope, D: Data> {
    fn end_condition<C: ConvergenceCheck<D> + 'static>(
        &self,
        check: C,
    ) -> (
        Stream<S, AbomonableArray2<D>>,
        Stream<S, AbomonableArray2<D>>,
    );
}

impl<S: Scope<Timestamp = Product<T, usize>>, T: Timestamp, D: Data + Display> EndCondition<S, D>
    for Stream<S, AbomonableArray2<D>>
{
    fn end_condition<C: ConvergenceCheck<D> + 'static>(
        &self,
        check: C,
    ) -> (
        Stream<S, AbomonableArray2<D>>,
        Stream<S, AbomonableArray2<D>>,
    ) {
        let worker = self.scope().index();
        let mut outputs = self.unary_frontier(Pipeline, "CheckConvergence", |_| {
            let mut iteration_count = 0;
            let mut centroid_stash: HashMap<_, AbomonableArray2<D>> = HashMap::new();

            move |input, output| {
                input.for_each(|cap, data| {
                    assert_eq!(data.len(), 1);

                    for new_centroids in data.drain(..) {
                        let done = if let Some(previous_centroids) = centroid_stash.remove(&cap) {
                            debug!("Checking convergence on worker {}", worker);
                            check.converges(
                                &previous_centroids.view(),
                                &new_centroids.view(),
                                iteration_count,
                            )
                        } else {
                            false
                        };

                        if done {
                            debug!("DONE!\n{}", new_centroids.view());
                        } else {
                            iteration_count += 1;
                            debug!("Continue to iteration {}", iteration_count);

                            // Re-Insert the current set of centroids into the stash
                            // with the timestamp for the next iteration
                            let delayed_cap =
                                cap.delayed(&Product::new(cap.outer.clone(), cap.inner + 1));
                            centroid_stash.insert(delayed_cap.clone(), new_centroids.clone());
                        }

                        output.session(&cap).give((done, new_centroids));
                    }
                });
            }
        })
        // split the centroids off into a separate stream (out of the loop) if the computation is done
        .partition(2, |(done, centroids)| {
            if done { (1, centroids) } else { (0, centroids) }
        });

        (outputs.pop().unwrap(), outputs.pop().unwrap())
    }
}
