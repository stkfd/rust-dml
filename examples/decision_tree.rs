extern crate flexi_logger;
extern crate log;
extern crate ml_dataflow;
extern crate ndarray;
extern crate timely;
extern crate timely_communication;

use flexi_logger::Logger;
use ml_dataflow::models::spdt::impurity::Gini;
use ml_dataflow::models::spdt::*;
use ml_dataflow::models::StreamingSupModel;
use ml_dataflow::models::TrainingData;
use ndarray::prelude::*;
use timely::dataflow::operators::*;
use timely_communication::initialize::Configuration;

fn main() {
    Logger::with_env_or_str("ml_dataflow=debug")
        .start()
        .unwrap();
    ::timely::execute(Configuration::Process(2), move |root| {
        let x = arr2(&[
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [1., 0., 2., 0., 3.],
            [1., 0., 2., 0., 3.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [1., 2., 3., 4., 5.],
            [1., 2., 3., 4., 5.],
        ]);

        let y: Array1<usize> = arr1(&[0, 0, 2, 2, 1, 1, 1, 1, 3, 3]);

        let mut model = StreamingDecisionTree::<Gini>::new(5, 500_000, 10);

        root.dataflow::<usize, _, _>(|scope| {
            let training_stream = vec![
                TrainingData {
                    x: x.clone().into(),
                    y: y.clone().into(),
                },
                TrainingData {
                    x: x.clone().into(),
                    y: y.clone().into(),
                },
            ].to_stream(scope);
            model
                .train(scope, training_stream)
                .expect("Training model")
                .inspect(|x| println!("Results: {:?}", x));
        });
        while root.step() {}
    }).expect("Execute dataflow");
}
