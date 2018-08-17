use data::providers::DataSource;
use data::serialization::*;
use failure::Error;
use fnv::FnvHashMap;
use ndarray::prelude::*;
use ndarray::Slice;
use ndarray_linalg::RealScalar;
use num_traits::Zero;
use rand::Rng;
use std::ops::{Mul, Sub};
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::{
    operators::{Exchange, Operator},
    Scope, Stream,
};
use timely::Data;
use timely::ExchangeData;

pub trait KMeansInitializer<T: Data> {
    fn select_initial_centroids<D: DataSource<AbomonableArray2<T>>>(
        data: &mut D,
        n_centroids: usize,
        cols: usize,
    ) -> Result<AbomonableArray2<T>, Error>;
}

pub trait KMeansStreamInitializer<T: Data> {
    fn select_initial_centroids<S: Scope>(
        samples: &Stream<S, AbomonableArray2<T>>,
        n_centroids: usize,
    ) -> Stream<S, AbomonableArray2<T>>;
}

#[derive(Clone, Copy, Abomonation)]
pub struct RandomSample;

impl<T: Data> KMeansInitializer<T> for RandomSample {
    fn select_initial_centroids<D: DataSource<AbomonableArray2<T>>>(
        data: &mut D,
        n_centroids: usize,
        _cols: usize,
    ) -> Result<AbomonableArray2<T>, Error> {
        let count = data.count()?;
        let mut rng = ::rand::thread_rng();

        let indices: Vec<usize> = (0..n_centroids)
            .map(|_| rng.gen_range(0 as usize, count))
            .collect();

        data.select(indices.as_slice())
    }
}

impl<T: ExchangeData + Copy + Zero> KMeansStreamInitializer<T> for RandomSample {
    fn select_initial_centroids<S: Scope>(
        samples: &Stream<S, AbomonableArray2<T>>,
        n_centroids: usize,
    ) -> Stream<S, AbomonableArray2<T>> {
        let centroids_per_peer = (n_centroids / samples.scope().peers()) + 1;
        samples
            .unary(Pipeline, "SelectInitialSamples", |_default_cap, _| {
                let mut count = centroids_per_peer;
                move |input, output| {
                    if count > 0 {
                        input.for_each(|time, data| {
                            let mut rng = ::rand::thread_rng();
                            for datum in data.iter() {
                                let matrix_view = datum.view();
                                while count > 0 {
                                    let indices: Vec<usize> = (0..centroids_per_peer
                                        .min(matrix_view.rows()))
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
            .exchange(|_| 0_u64)
            .unary(Pipeline, "AggregateSelectedCentroids", |_, _| {
                let mut stash = FnvHashMap::default();
                move |input, output| {
                    input.for_each(|cap_ref, data| {
                        let cap = cap_ref.retain();
                        for peer_selection in data.drain(..) {
                            let partial_centroids = peer_selection.view();
                            let (received, centroids) =
                                stash.entry(cap.clone()).or_insert_with(|| {
                                    (
                                        0,
                                        Some(Array2::<T>::zeros((
                                            n_centroids,
                                            partial_centroids.cols(),
                                        ))),
                                    )
                                });
                            let centroids = centroids.as_mut().unwrap();

                            if *received < centroids.rows() {
                                let range_end =
                                    <usize>::min(*received + partial_centroids.rows(), n_centroids);
                                println!("{:?}", *received..range_end);
                                centroids
                                    .slice_mut(s![*received..range_end, ..])
                                    .assign(&partial_centroids);
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

pub struct KMeansPlusPlus;

impl<T: Data + RealScalar + Sub<T> + Mul<T>> KMeansInitializer<T> for KMeansPlusPlus {
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
