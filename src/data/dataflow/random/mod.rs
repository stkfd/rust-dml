#![allow(dead_code)]

use data::dataflow::random::params::ToDistribution;
use data::TrainingData;
use ndarray::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::marker::PhantomData;
use timely::dataflow::operators::generic::source;
use timely::dataflow::Scope;
use timely::dataflow::Stream;
use timely::Data;

mod classification;
mod regression;

pub use self::classification::RandClassificationTrainingSource;
pub use self::regression::RandRegressionTrainingSource;

/// Configuration for random distributions
pub mod params {
    use rand::distributions::uniform::SampleUniform;
    use rand::distributions::{Distribution, Normal, Uniform};
    use rand::Rng;

    /// Trait for a struct that can generate an instance of an associated
    /// `Distribution<T>`
    pub trait ToDistribution<T> {
        type Dist: Distribution<T>;

        /// Create a `Distribution` instance from this configuration
        fn to_distribution(&self) -> Self::Dist;
    }

    /// Parameters for a normal random distribution (mean & standard deviation)
    #[derive(Clone)]
    pub struct NormalParams {
        pub mean: f64,
        pub std_dev: f64,
    }

    impl NormalParams {
        pub fn new(mean: f64, std_dev: f64) -> Self {
            NormalParams { mean, std_dev }
        }
    }

    impl<'a, T> ToDistribution<T> for NormalParams
    where
        Normal: Distribution<T>,
    {
        type Dist = Normal;

        fn to_distribution(&self) -> Self::Dist {
            Normal::new(self.mean, self.std_dev)
        }
    }

    /// Parameters for a uniform random distribution (upper & lower bounds)
    #[derive(Clone)]
    pub struct UniformParams<T> {
        pub low: T,
        pub high: T,
    }

    impl<T> UniformParams<T> {
        pub fn new(low: T, high: T) -> Self {
            UniformParams { low, high }
        }
    }

    impl<'a, T> ToDistribution<T> for UniformParams<T>
    where
        T: SampleUniform + Clone + 'a,
    {
        type Dist = Uniform<T>;

        fn to_distribution(&self) -> Self::Dist {
            Uniform::new(self.low.clone(), self.high.clone())
        }
    }

    #[derive(Clone)]
    pub struct DummyDistribution<T>(pub T);
    impl<T: Clone> Distribution<T> for DummyDistribution<T> {
        fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> T {
            self.0.clone()
        }
    }

    impl<T: Clone> ToDistribution<T> for DummyDistribution<T> {
        type Dist = Self;
        fn to_distribution(&self) -> Self::Dist {
            self.clone()
        }
    }
}

#[cfg(test)]
mod test {
    use super::params::*;
    use super::*;
    use timely::dataflow::operators::capture::Extract;
    use timely::dataflow::operators::*;
    use timely::progress::nested::Summary;
    use timely::progress::timestamp::RootTimestamp;

    #[test]
    fn random_source_f64() {
        let source = RandClassificationTrainingSource::<f64, f64, _, _>::default()
            .samples(200, 2, 3)
            .x_distributions(arr2(&[
                [
                    NormalParams::new(1., 1.),
                    NormalParams::new(1., 1.),
                    NormalParams::new(1., 1.),
                ],
                [
                    NormalParams::new(2., 1.),
                    NormalParams::new(2., 1.),
                    NormalParams::new(2., 1.),
                ],
                [
                    NormalParams::new(4., 1.),
                    NormalParams::new(4., 1.),
                    NormalParams::new(4., 1.),
                ],
            ]))
            .y_distributions(arr1(&[
                NormalParams::new(2., 0.5),
                NormalParams::new(4., 0.5),
                NormalParams::new(0., 0.5),
            ]));

        let result = ::timely::example(move |scope| {
            source
                .to_stream(Summary::Local(1), RootTimestamp::new(1), scope)
                .inspect(|data: &TrainingData<f64, f64>| {
                    assert_eq!(data.x().dim(), (200, 3));
                    assert_eq!(data.y().dim(), 200);
                })
                .count()
                .inspect(|&count| assert_eq!(2, count))
                .capture()
        }).extract();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn random_source_regression() {
        let source = RandRegressionTrainingSource::new(
            |x: &ArrayView1<f64>, x_mapped: &mut ArrayViewMut1<f64>| {
                x_mapped.assign(x);
                x.iter().sum()
            },
        ).samples(200, 2)
            .x_distributions(arr1(&[
                NormalParams::new(0., 1.),
                NormalParams::new(5., 1.),
                NormalParams::new(10., 1.),
            ]));

        let result = ::timely::example(move |scope| {
            source
                .to_stream(Summary::Local(1), RootTimestamp::new(3), scope)
                .inspect(|data: &TrainingData<f64, f64>| {
                    assert_eq!(data.x().dim(), (200, 3));
                    assert_eq!(data.y().dim(), 200);
                })
                .count()
                .inspect(|&count| assert_eq!(2, count))
                .capture()
        }).extract();
        assert_eq!(result.len(), 3);
    }
}
