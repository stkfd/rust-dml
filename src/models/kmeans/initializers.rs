use data::serialization::*;
use fnv::FnvHashMap;
use ndarray::prelude::*;
use ndarray_linalg::into_scalar;
use ndarray_linalg::RealScalar;
use num_traits::Num;
use num_traits::Zero;
use rand::Rng;
use std::cmp::Ordering;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::scopes::Child;
use timely::dataflow::{operators::*, Scope, Stream};
use timely::progress::Timestamp;
use timely::{Data, ExchangeData};

pub trait KMeansInitializer<T: Data> {
    fn select_initial_centroids<S: Scope>(
        samples: &Stream<S, AbomonableArray2<T>>,
        n_centroids: usize,
    ) -> Stream<S, AbomonableArray2<T>>;
}

#[derive(Clone, Copy, Abomonation)]
pub struct RandomSample;

impl<T: ExchangeData + Copy + Num + RealScalar> KMeansInitializer<T> for RandomSample {
    fn select_initial_centroids<S: Scope>(
        samples: &Stream<S, AbomonableArray2<T>>,
        n_centroids: usize,
    ) -> Stream<S, AbomonableArray2<T>> {
        let centroids_per_peer = (n_centroids / samples.scope().peers()) + 1;
        samples
            .select_random_samples_uniform(centroids_per_peer)
            .exchange(|_| 0_u64)
            .aggregate_centroids(n_centroids)
    }
}

pub struct KMeansPlusPlus;

impl<T: ExchangeData + Copy + Num + RealScalar> KMeansInitializer<T> for KMeansPlusPlus {
    fn select_initial_centroids<S: Scope>(
        samples: &Stream<S, AbomonableArray2<T>>,
        n_centroids: usize,
    ) -> Stream<S, AbomonableArray2<T>> {
        let mut scope = samples.scope();
        let worker = scope.index();
        let peers = scope.peers();

        let first_centroid = samples
            .filter(move |_| worker == 0)
            .select_random_samples_uniform(1);

        scope.scoped(|loop_scope| {
            let (loop_handle, loop_stream) = loop_scope.loop_variable(n_centroids, 1);
            let (next_iter, finished) = first_centroid
                .enter(loop_scope)
                .concat(&loop_stream)
                .exchange(move |_| ::rand::thread_rng().gen_range(0, peers as u64))
                .select_random_distance_weighted(&samples.enter(loop_scope))
                .branch_when(move |time| time.inner >= n_centroids);
            next_iter.connect_loop(loop_handle);
            finished.leave()
        })
    }
}

trait SelectRandomSamplesUniform<S: Scope, T: Data> {
    fn select_random_samples_uniform(&self, per_peer: usize) -> Stream<S, AbomonableArray2<T>>;
}

impl<S: Scope, T: Data + Copy> SelectRandomSamplesUniform<S, T> for Stream<S, AbomonableArray2<T>> {
    fn select_random_samples_uniform(&self, per_peer: usize) -> Stream<S, AbomonableArray2<T>> {
        self.unary(Pipeline, "SelectRandomSamples", |_, _| {
            let mut count = per_peer;
            move |input, output| {
                if count > 0 {
                    input.for_each(|time, data| {
                        let mut rng = ::rand::thread_rng();
                        for datum in data.iter() {
                            let matrix_view = datum.view();
                            while count > 0 {
                                let indices: Vec<usize> = (0..per_peer.min(matrix_view.rows()))
                                    .map(|_| rng.gen_range(0_usize, matrix_view.rows()))
                                    .collect();
                                count -= indices.len();
                                let final_centroids: AbomonableArray2<_> = matrix_view.select(Axis(0), &indices).into();
                                output.session(&time).give(final_centroids);
                            }
                        }
                    });
                }
            }
        })
    }
}

trait AggregateCentroids<S: Scope, T: Data> {
    fn aggregate_centroids(&self, n_centroids: usize) -> Stream<S, AbomonableArray2<T>>;
}

impl<S: Scope, T: Data + Copy + Num> AggregateCentroids<S, T> for Stream<S, AbomonableArray2<T>> {
    fn aggregate_centroids(&self, n_centroids: usize) -> Stream<S, AbomonableArray2<T>> {
        self.unary(Pipeline, "AggregateSelectedCentroids", |_, _| {
            let mut stash = FnvHashMap::default();
            move |input, output| {
                input.for_each(|cap_ref, data| {
                    let cap = cap_ref.retain();
                    for peer_selection in data.drain(..) {
                        let partial_centroids = peer_selection.view();
                        let (received, centroids) = stash.entry(cap.clone()).or_insert_with(|| {
                            (
                                0,
                                Some(Array2::<T>::zeros((n_centroids, partial_centroids.cols()))),
                            )
                        });
                        let centroids = centroids.as_mut().unwrap();

                        if *received < centroids.rows() {
                            let range_end =
                                <usize>::min(*received + partial_centroids.rows(), n_centroids);
                            centroids
                                .slice_mut(s![*received..range_end, ..])
                                .assign(&partial_centroids.slice(s![0..(range_end - *received), ..]));
                        }
                        *received += partial_centroids.rows();
                    }
                });

                stash.retain(|time, (received, centroids)| {
                    let full = *received >= centroids.as_ref().unwrap().rows();
                    if full {
                        output.session(&time).give(centroids.take().unwrap().into());
                    }
                    !full
                });
            }
        })
    }
}

trait SelectRandomDistanceWeighted<'a, S: Scope, Ts: Timestamp, T: Data> {
    fn select_random_distance_weighted(
        &self,
        samples: &Stream<Child<'a, S, Ts>, AbomonableArray2<T>>,
    ) -> Self;
}

impl<'a, S: Scope, T: Data + Num + RealScalar, Ts: Timestamp>
    SelectRandomDistanceWeighted<'a, S, Ts, T> for Stream<Child<'a, S, Ts>, AbomonableArray2<T>>
{
    fn select_random_distance_weighted(
        &self,
        samples: &Stream<Child<'a, S, Ts>, AbomonableArray2<T>>,
    ) -> Stream<Child<'a, S, Ts>, AbomonableArray2<T>> {
        self.binary_frontier(
            samples,
            Pipeline,
            Pipeline,
            "SelectWeightedCentroid",
            |_, _| {
                let mut samples_stash = FnvHashMap::default();
                let mut centroids_stash = FnvHashMap::default();
                move |centroids_input, samples_input, output| {
                    samples_input.for_each(|cap_ref, data| {
                        samples_stash
                            .entry(cap_ref.time().outer.clone())
                            .or_insert_with(Vec::new)
                            .extend(data.drain(..))
                    });

                    centroids_input.for_each(|cap_ref, data| {
                        centroids_stash
                            .entry(cap_ref.retain())
                            .or_insert_with(Vec::new)
                            .extend(data.drain(..))
                    });

                    centroids_stash.retain(|cap, centroids_vec| {
                        let centroids_received = !centroids_input.frontier().less_equal(&cap);

                        if centroids_received {
                            let centroids = centroids_vec[0].view();
                            let mut extended_centroids =
                                Array2::zeros((centroids.rows() + 1, centroids.cols()));
                            extended_centroids.assign(&centroids);
                            let samples = samples_stash.get(&cap.time().outer).unwrap();

                            let mut distances: Vec<_> = samples
                                .iter()
                                .map(|samples_chunk| {
                                    samples_chunk
                                        .view()
                                        .map_axis(Axis(0), |point| {
                                            centroids
                                                .outer_iter()
                                                .map(|centroid| {
                                                    (&centroid - &point)
                                                        .iter()
                                                        .map(|x| x.abs_sqr())
                                                        .sum::<T::Real>()
                                                })
                                                .min_by(|a, b| {
                                                    a.partial_cmp(&b)
                                                        .unwrap_or(::std::cmp::Ordering::Less)
                                                })
                                                .unwrap()
                                        })
                                        .into_raw_vec()
                                })
                                .collect();

                            let mut cumulative_distance = T::Real::zero();
                            for chunk in &mut distances {
                                for distance in chunk {
                                    cumulative_distance = cumulative_distance + *distance;
                                    *distance = cumulative_distance;
                                }
                            }

                            let rand_num: T::Real =
                                into_scalar::<T::Real>(::rand::thread_rng().gen())
                                    * cumulative_distance;

                            for (chunk_idx, chunk) in distances.iter().enumerate() {
                                let index = match chunk.binary_search_by(|probe| {
                                    probe.partial_cmp(&rand_num).unwrap_or(Ordering::Less)
                                }) {
                                    Ok(found) => found,
                                    Err(insert_at) => insert_at,
                                };

                                if index < chunk.len() {
                                    let last_row = extended_centroids.rows() - 1;
                                    extended_centroids
                                        .row_mut(last_row)
                                        .assign(&samples[chunk_idx].view().row(index));
                                    break;
                                }
                            }
                            output.session(&cap).give(extended_centroids.into());
                        }

                        !centroids_received
                    });
                }
            },
        )
    }
}

/*impl<T: Data + RealScalar + Sub<T> + Mul<T>> KMeansInitializer<T> for KMeansPlusPlus {
    fn select_initial_centroids<D: DataSource<AbomonableArray2<T>>>(
        data: &mut D,
        n_centroids: usize,
        cols: usize,
    ) -> Result<AbomonableArray2<T>, Error> {
        let count = data.count()?;
        let mut rng = ::rand::thread_rng();

        let mut centroids = Array2::zeros([n_centroids, cols]);

        let first: Array2<_> = data.select(&[rng.gen_range(0usize, count)])?.into();
        centroids
            .subview_mut(Axis(0), 0)
            .assign(&first.into_subview(Axis(0), 0));

        for i in 1..n_centroids {
            let selected_centroids = centroids.slice_axis(Axis(0), Slice::from(0..(i - 1)));

            // get all squared distances from each point to the nearest already chosen centroid
            let mut distances = Array1::zeros(count);
            data.chunk_indices(5000)?
                .map(|slice_index| {
                    (
                        Slice::from(slice_index.start..(slice_index.start + slice_index.length)),
                        data.slice(slice_index).unwrap().into(),
                    )
                })
                .map(|(slice, chunk): (_, Array2<T>)| {
                    let distances_chunk = chunk.map_axis(Axis(0), |point| {
                        selected_centroids
                            .outer_iter()
                            .map(|centroid| {
                                (&centroid - &point)
                                    .iter()
                                    .map(|x| x.abs_sqr())
                                    .sum::<T::Real>()
                            })
                            .min_by(|a, b| a.partial_cmp(&b).unwrap_or(::std::cmp::Ordering::Less))
                            .unwrap()
                    });
                    (slice, distances_chunk)
                })
                .for_each(|(slice, distances_chunk)| {
                    distances
                        .slice_axis_mut(Axis(0), slice)
                        .assign(&distances_chunk)
                });

            // TODO: Redo weighted random sampling using generic scalars
        }

        Ok(centroids.into())
    }
}
*/
