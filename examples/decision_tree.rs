extern crate flexi_logger;
extern crate log;
extern crate ml_dataflow;
extern crate ndarray;
extern crate timely;
extern crate timely_communication;

use flexi_logger::Logger;
use ml_dataflow::data::serialization::{AbomonableArray, AsView};
use ml_dataflow::models::decision_tree::classification::impurity::Gini;
use ml_dataflow::models::decision_tree::classification::*;
use ml_dataflow::models::StreamingSupModel;
use ml_dataflow::data::TrainingData;
use ndarray::prelude::*;
use timely::dataflow::operators::*;
use timely_communication::initialize::Configuration;

fn main() {
    Logger::with_env_or_str("ml_dataflow=debug")
        .start()
        .unwrap();
    ::timely::execute(Configuration::Process(2), move |root| {
        let x = arr2(&[
            [10., 5., 10., 10., 50.],
            [10., 5., 10., 10., 50.],
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

        let mut model = StreamingClassificationTree::<Gini, _, _>::new(5, 500_000, 5);

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
            let trees = model
                .train(scope, training_stream)
                .expect("Training model")
                .inspect(|x| println!("Results: {:?}", x));
            model
                .predict(
                    trees,
                    vec![AbomonableArray::from(x.clone())].to_stream(scope),
                )
                .unwrap()
                .inspect(|d| println!("{}", d.view()));
        });
        while root.step() {}
    }).expect("Execute dataflow");
}
