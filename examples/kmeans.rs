extern crate flexi_logger;
extern crate log;
extern crate ml_dataflow;
extern crate ndarray;
extern crate timely;
extern crate timely_communication;

use flexi_logger::Logger;
use ml_dataflow::data::serialization::*;
use ml_dataflow::models::kmeans::{
    initializers::RandomSample, ConvergenceCriteria, KmeansStreaming,
};
use ml_dataflow::models::{Predict, Train};
use ndarray::prelude::*;
use timely::dataflow::operators::*;
use timely_communication::initialize::Configuration;

fn main() {
    Logger::with_env_or_str("ml_dataflow=debug")
        .start()
        .unwrap();
    ::timely::execute(Configuration::Process(2), move |root| {
        let n_clusters = 2;
        let some_data: AbomonableArray<_, _> = arr2(&[
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
        ]).into();

        let end_criteria = <ConvergenceCriteria<f64>>::default().limit_iterations(5);
        let model = <KmeansStreaming<f64, RandomSample>>::new(
            n_clusters,
            some_data.view().cols(),
            end_criteria,
        );

        root.dataflow::<usize, _, _>(|scope| {
            let training_result = vec![some_data.clone()]
                .to_stream(scope)
                .train(&model)
                .inspect(|result| println!("{:?}", result.view()));

            vec![some_data]
                .to_stream(scope)
                .predict(&model, training_result)
                .inspect(|result| println!("{:?}", result.as_ref().unwrap().view()));
        });

        while root.step() {}
    }).expect("Execute dataflow");
}
