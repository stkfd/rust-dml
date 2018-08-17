//! The K-Means Clustering Algorithm.

use self::aggregator::*;
use self::assign_points::AssignPoints;
pub use self::convergence::*;
pub use self::initializers::KMeansInitializer;
use self::stop_condition::StopCondition;
use data::dataflow::{AsyncResult, IndexDataStream};
use data::providers::{DataSource, DataSourceSpec};
use data::serialization::*;
use data::TrainingData;
use failure::Error;
use models::kmeans::initializers::KMeansStreamInitializer;
use models::ModelAttributes;
use models::SupModelAttributes;
use models::Train;
use models::UnSupModel;
use ndarray::indices;
use ndarray::prelude::*;
use ndarray::{NdProducer, ScalarOperand, Zip};
use ndarray_linalg::{Norm, Scalar};
use num_traits::cast::FromPrimitive;
use num_traits::Float;
use num_traits::NumAssignOps;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::mpsc;
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

const CHUNK_SIZE: usize = 5000;

pub struct Kmeans<Item: Data, Init: KMeansInitializer<Item>> {
    n_clusters: usize,
    cols: usize,
    centroids: AsyncResult<AbomonableArray2<Item>>,
    end_criteria: ConvergenceCriteria<Item>,
    phantom_data: PhantomData<Init>,
}

#[derive(Abomonation, Clone)]
pub struct KmeansStreaming<Item: Data, Init: KMeansStreamInitializer<Item> + Data> {
    n_clusters: usize,
    cols: usize,
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
    Item: ExchangeData + FromPrimitive + Float + Display + ScalarOperand + Scalar + NumAssignOps,
    Init: KMeansInitializer<Item>,
{
    fn predict<S: Scope, Sp: DataSourceSpec<AbomonableArray2<Item>>>(
        &mut self,
        scope: &mut S,
        inputs: Sp,
    ) -> Result<Stream<S, AbomonableArray2<usize>>, Error> {
        if scope.index() != 0 {
            return Ok(vec![].to_stream(scope));
        }
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
    ) -> Result<Stream<S, AbomonableArray2<Item>>, Error> {
        let n_clusters = self.n_clusters;
        let cols = self.cols;

        let end_criteria = self.end_criteria.clone();
        let max_iterations = self
            .end_criteria
            .max_iterations
            .unwrap_or(<usize>::max_value());

        let mut provider = input_spec.to_provider()?;
        let initial_centroids: Vec<AbomonableArray2<Item>> = vec![Init::select_initial_centroids(
            &mut provider,
            n_clusters,
            cols,
        )?];

        debug!(
            "Selected initial centroids: {}",
            initial_centroids[0].view()
        );

        let (result_sender, result_receiver) = mpsc::channel();
        if scope.index() == 0 {
            self.centroids = AsyncResult::Receiver(result_receiver);
        }

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
                    (slice_index, provider.slice(slice_index).unwrap())
                });

            let (loop_handle, loop_stream) = inner.loop_variable(max_iterations, 1);

            let (done, next_iteration) = initial_centroids
                .to_stream(inner)
                .filter(move |_| worker_index == 0)
                .concat(&loop_stream)
                // checks whether the convergence criteria are met and aborts the loop
                .stop_condition(end_criteria);

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

impl<T: ExchangeData, Init: ExchangeData + KMeansStreamInitializer<T>> ModelAttributes
    for KmeansStreaming<T, Init>
{
    type UnlabeledSamples = AbomonableArray2<T>;
    type TrainingResult = AbomonableArray2<T>;
}

impl<T: ExchangeData, Init: ExchangeData + KMeansStreamInitializer<T>> SupModelAttributes
    for KmeansStreaming<T, Init>
{
    type LabeledSamples = TrainingData<T, T>;
    type Predictions = AbomonableArray1<T>;
    type PredictErr = KMeansError;
}

#[derive(Fail, Debug, Abomonation, Clone)]
pub enum KMeansError {
    #[fail(display = "Unknown Error")]
    Unknown,
}

impl<S, T, Init> Train<S, KmeansStreaming<T, Init>> for Stream<S, AbomonableArray2<T>>
where
    S: Scope,
    T: ExchangeData + Scalar + NumAssignOps + ScalarOperand + Float + Debug + FromPrimitive,
    Init: ExchangeData + KMeansStreamInitializer<T>,
{
    fn train(&self, model: &KmeansStreaming<T, Init>) -> Stream<S, AbomonableArray2<T>> {
        let n_clusters = model.n_clusters;
        let cols = model.cols;

        let end_criteria = model.end_criteria.clone();
        let max_iterations = model
            .end_criteria
            .max_iterations
            .unwrap_or(<usize>::max_value());

        let initial_centroids =
            Init::select_initial_centroids(self, n_clusters).inspect(|initial_centroids| {
                debug!("Selected initial centroids: {:?}", initial_centroids.view());
            });

        self.scope().scoped(|inner| {
            let worker_index = inner.index();
            debug!("Constructing worker {}", inner.index());

            let (loop_handle, loop_stream) = inner.loop_variable(max_iterations, 1);

            let (done, next_iteration) = initial_centroids
                .enter(inner)
                .filter(move |_| worker_index == 0)
                .concat(&loop_stream)
                // checks whether the convergence criteria are met and aborts the loop
                .stop_condition(end_criteria);

            next_iteration
                .broadcast()
                .assign_points(&(self.index_data().enter(inner)))
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
                debug!("Finished: {:?}", c.view());
            }).leave()
        })
    }
}
