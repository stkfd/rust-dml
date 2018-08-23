//! The K-Means Clustering Algorithm.

use self::aggregator::*;
use self::assign_points::AssignPoints;
pub use self::convergence::*;
pub use self::initializers::KMeansInitializer;
use self::stop_condition::StopCondition;
use data::dataflow::{ApplyLatest, IndexDataStream};
use data::serialization::*;
use models::kmeans::initializers::KMeansStreamInitializer;
use models::*;
use ndarray::indices;
use ndarray::prelude::*;
use ndarray::{NdProducer, ScalarOperand, Zip};
use ndarray_linalg::{Norm, Scalar};
use num_traits::{cast::FromPrimitive, Float, NumAssignOps};
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use timely::dataflow::channels::pact::Pipeline;
use timely::progress::Timestamp;
use timely::{
    dataflow::{operators::*, Scope, Stream},
    progress::nested::product::Product,
    Data, ExchangeData,
};

mod aggregator;
mod assign_points;
mod convergence;
pub mod initializers;
mod stop_condition;

#[derive(Abomonation, Clone)]
pub struct Kmeans<Item: Data, Init: KMeansStreamInitializer<Item> + Data> {
    n_clusters: usize,
    cols: usize,
    end_criteria: ConvergenceCriteria<Item>,
    phantom_data: PhantomData<Init>,
}

impl<Item: Data, Init: Data + KMeansStreamInitializer<Item>> Kmeans<Item, Init> {
    pub fn new(n_clusters: usize, cols: usize, end_criteria: ConvergenceCriteria<Item>) -> Self {
        Kmeans {
            n_clusters,
            cols,
            end_criteria,
            phantom_data: PhantomData,
        }
    }
}

impl<T: ExchangeData, Init: ExchangeData + KMeansStreamInitializer<T>> ModelAttributes
    for Kmeans<T, Init>
{
    type TrainingResult = AbomonableArray2<T>;
}

impl<T: ExchangeData, Init: ExchangeData + KMeansStreamInitializer<T>> LabelingModelAttributes
    for Kmeans<T, Init>
{
    type Predictions = AbomonableArray2<usize>;
    type PredictErr = KMeansError;
}

#[derive(Fail, Debug, Abomonation, Clone)]
pub enum KMeansError {
    #[fail(display = "Unknown Error")]
    Unknown,
}

impl<S, T, Init> Train<S, Kmeans<T, Init>> for Stream<S, AbomonableArray2<T>>
where
    S: Scope,
    T: ExchangeData + Scalar + NumAssignOps + ScalarOperand + Float + Debug + FromPrimitive,
    Init: ExchangeData + KMeansStreamInitializer<T>,
{
    fn train(&self, model: &Kmeans<T, Init>) -> Stream<S, AbomonableArray2<T>> {
        let n_clusters = model.n_clusters;
        let cols = model.cols;

        let end_criteria = model.end_criteria.clone();
        let max_iterations = model
            .end_criteria
            .max_iterations
            .unwrap_or(<usize>::max_value());

        let initial_centroids = Init::select_initial_centroids(self, n_clusters);

        initial_centroids.inspect(|initial_centroids| {
            debug!("Selected initial centroids: {:?}", initial_centroids.view());
        });

        self.scope().scoped(|inner_scope| {
            let worker_index = inner_scope.index();
            debug!("Constructing worker {}", inner_scope.index());

            let (loop_handle, loop_stream) = inner_scope.loop_variable(max_iterations, 1);

            let (done, next_iteration) = initial_centroids
                .enter(inner_scope)
                .concat(&loop_stream)
                // checks whether the convergence criteria are met and aborts the loop
                .stop_condition(end_criteria);

            next_iteration
                .broadcast()
                .assign_points(&(self.index_data().enter(inner_scope)))
                .exchange(|_| 0u64)
                .accumulate_statistics(AggregationStatistics::new(n_clusters, cols))
                .map(move |stats| {
                    debug!(
                        "Worker {}: Aggregated all assignments, calculating new centroids",
                        worker_index
                    );
                    // re-estimate centroids
                    AggregationStatistics::from(stats)
                        .centroid_estimate()
                        .into()
                })
                .connect_loop(loop_handle);

            done.inspect(move |c| {
                debug!("worker {}", worker_index);
                debug!("Finished: {:?}", c.view());
            }).leave()
        })
    }
}

impl<S, T, Init> Predict<S, Kmeans<T, Init>, KMeansError> for Stream<S, AbomonableArray2<T>>
where
    S: Scope,
    T: ExchangeData + Scalar + NumAssignOps + ScalarOperand + Float + Debug + FromPrimitive,
    Init: ExchangeData + KMeansStreamInitializer<T>,
{
    fn predict(
        &self,
        _model: &Kmeans<T, Init>,
        train_results: Stream<S, AbomonableArray2<T>>,
    ) -> Stream<S, Result<AbomonableArray2<usize>, ModelError<KMeansError>>> {
        train_results.apply_latest(self, |_time, centroids, samples| {
            let samples = samples.view();
            let centroids = centroids.view();

            let mut assignments = unsafe { Array2::<usize>::uninitialized((samples.rows(), 2)) };
            Zip::from(assignments.genrows_mut())
                .and(samples.genrows())
                .and(indices(samples.genrows().raw_dim()))
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
            Ok(assignments.into())
        })
    }
}
