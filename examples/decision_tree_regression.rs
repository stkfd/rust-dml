extern crate flexi_logger;
extern crate log;
extern crate ml_dataflow;
extern crate ndarray;
extern crate timely;
extern crate timely_communication;

use flexi_logger::Logger;
use ml_dataflow::data::serialization::{AbomonableArray, AsView};
use ml_dataflow::models::decision_tree::regression::*;
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
            [0],
            [0],
            [1],
            [1],
            [2],
            [2],
            [3],
            [3],
        ]);

        let y: Array1<f64> = arr1(&[10., 10., 12., 12., 14., 14., 16., 16.]);

        let mut model = StreamingRegressionTree::new(2, 500_000, 5, 1.0);

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
