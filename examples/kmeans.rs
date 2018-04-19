extern crate ml_dataflow;
extern crate ndarray;
extern crate timely;
extern crate timely_communication;

use ml_dataflow::{data::providers::array::ArrayProviderSpec,
                  models::{kmeans::{initializers::RandomSample, ConvergenceCriteria, Kmeans},
                           UnSupModel}};
use ndarray::prelude::*;
use timely_communication::initialize::Configuration;

fn main() {
    ::timely::execute(Configuration::Thread, move |root| {
        let n_clusters = 2;
        let some_data = arr2(&[
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
        ]);

        let end_criteria = <ConvergenceCriteria<f64>>::default().limit_iterations(5);
        let mut model =
            <Kmeans<f64, RandomSample>>::new(n_clusters, some_data.cols(), end_criteria);
        let source = ArrayProviderSpec::new(some_data);

        root.dataflow::<usize, _, _>(|scope| {
            model.train(scope, source.clone()).unwrap();
        });

        while root.step() {}

        root.dataflow::<usize, _, _>(|scope| {
            model.predict(scope, source.clone()).unwrap();
        });
    }).unwrap();
}
