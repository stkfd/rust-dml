extern crate ml_dataflow;
extern crate ndarray;
extern crate timely;
extern crate timely_communication;
extern crate log;
extern crate flexi_logger;

use ml_dataflow::data::providers::array::ArrayProviderSpec;
use ml_dataflow::models::kmeans::{ConvergenceCriteria, Kmeans, initializers::RandomSample};
use ml_dataflow::models::UnSupModel;
use ml_dataflow::data::serialization::AsView;
use ndarray::prelude::*;
use timely_communication::initialize::Configuration;
use timely::dataflow::operators::*;
use flexi_logger::Logger;

fn main() {
    Logger::with_env_or_str("ml_dataflow=debug").start().unwrap();
    ::timely::execute(Configuration::Process(2), move |root| {
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
            model.train(scope, source.clone()).expect("Training model");
        });

        while root.step() {}

        root.dataflow::<usize, _, _>(|scope| {        
                model
                    .predict(scope, source.clone()).expect("Making predictions for some data")
                    .inspect(|assignments| {
                        let array: ArrayView<_, _> = assignments.view();
                        for assignment in array.genrows() {
                            println!("{} -> {}", assignment[0], assignment[1]);
                        }
                    });
        });

        while root.step() {}
    }).expect("Execute dataflow");
}
