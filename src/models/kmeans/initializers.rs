use failure::Error;
use data::providers::DataSource;
use data::serialization::*;
use ndarray::prelude::*;
use ndarray::Slice;
use ndarray_linalg::{RealScalar};
use rand::Rng;
use std::ops::{Mul, Sub};
use timely::Data;

pub trait KMeansInitializer<T: Data> {
    fn select_initial_centroids<D: DataSource<AbomonableArray2<T>>>(
        data: &mut D,
        n_centroids: usize,
        cols: usize,
    ) -> Result<AbomonableArray2<T>, Error>;
}

pub struct RandomSample {}

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

pub struct KMeansPlusPlus {}

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
                        Slice::from(
                            slice_index.start..(slice_index.start + slice_index.length),
                        ),
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
                            .min_by(|a, b| {
                                a.partial_cmp(&b).unwrap_or(::std::cmp::Ordering::Less)
                            })
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
